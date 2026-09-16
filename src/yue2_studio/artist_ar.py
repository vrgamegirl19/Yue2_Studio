"""AR-only artist adapters, adapted from Mothersuperior's pinned v4 recipe.

No model loading or GPU work at import. The companion NAR is NOT trained here.
"""
import math
import unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file


class ArtistLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base=base
        self.down=nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)/math.sqrt(base.in_features))
        self.up=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))

    def forward(self,x):
        return self.base(x)+F.linear(F.linear(x.float(),self.down),self.up).to(x.dtype)


def place_ar(model,device):
    """Only the AR branch participates in loss(); keep unused NAR weights on CPU."""
    model.eval().requires_grad_(False).to('cpu')
    for module in (model.model.embed_tokens,model.model.rotary_emb,model.model.norm,model.lm_head):
        module.to(device)
    for layer in model.model.layers:
        for name in ('input_layernorm','self_attn','post_attention_layernorm','mlp'):
            getattr(layer,name).to(device)
    return model


def install(model,rank):
    model.eval().requires_grad_(False)
    adapters={}
    for i,layer in enumerate(model.model.layers):
        for branch,names in [('self_attn',('q_proj','k_proj','v_proj','o_proj')),('mlp',('gate_proj','up_proj','down_proj'))]:
            parent=getattr(layer,branch)
            for name in names:
                adapter=ArtistLinear(getattr(parent,name),rank)
                setattr(parent,name,adapter)
                adapters[f'model.layers.{i}.{branch}.{name}']=adapter
    if not adapters:raise ValueError('No AR layers found.')
    return adapters


def export(adapters,path,metadata):
    state={}
    for name,module in adapters.items():
        for suffix,value in [('lora_down',module.down),('lora_up',module.up)]:
            state[name+'.'+suffix+'.weight']=value.detach().cpu().contiguous()
    if not all(torch.isfinite(x).all() for x in state.values()):raise ValueError('Non-finite adapter; refusing export.')
    temporary=path.with_suffix('.tmp')
    save_file(state,str(temporary),metadata={**metadata,'format':'yue2-artist-ar-v1','scale':'1'})
    temporary.replace(path)


def sequence(item,device,max_length=12288):
    from yue2.protocol import CODEC_OFFSET,MUSIC_END
    codec=np.asarray(item['codec'])
    if codec.ndim!=1 or not len(codec) or codec.dtype.kind not in 'iu' or codec.min()<0 or codec.max()>=32768:
        raise ValueError('Invalid semantic tokens: '+item['name'])
    prefix=list(item['prefix'])
    if not prefix or len(prefix)+len(codec)+1>max_length:
        raise ValueError('Full song exceeds training context; no audio was silently truncated: '+item['name'])
    return torch.tensor([prefix+[int(c)+CODEC_OFFSET for c in codec]+[MUSIC_END]],device=device),len(prefix)


def cursor_targets(item,tokenizer,device):
    from yue2.protocol import INSTRUCTIONS
    words=np.asarray(item['words'])
    if words.ndim!=2 or words.shape[1]!=5 or not len(words) or not np.isfinite(words).all():
        raise ValueError('Missing or invalid lyric alignment: '+item['name'])
    head=f"{INSTRUCTIONS['off']}\n[Tags]\n{item['style']}\n[Lyrics]\n"
    text=head+item['lyrics']+'\n'
    ids_full=tokenizer.encode(text)
    if list(item['prefix'][1:len(ids_full)+1])!=ids_full:
        raise ValueError('Cannot map lyric tokens; alignment was not silently disabled.')
    # BPE can merge the header's newline with leading lyric whitespace. Map the
    # actual complete prefix in UTF-8 bytes rather than separately tokenizing its
    # header. Byte spans also handle tokens that split a Unicode character.
    enc=getattr(tokenizer,'_enc',None)
    pieces=[enc.decode_single_token_bytes(i) if enc is not None else tokenizer.decode([i]).encode('utf-8') for i in ids_full]
    if b''.join(pieces)!=unicodedata.normalize('NFC',text).encode('utf-8'):
        raise ValueError('Tokenizer byte spans do not match the training text.')
    all_ends=np.cumsum([len(piece) for piece in pieces]).tolist()
    header_bytes=len(unicodedata.normalize('NFC',head).encode('utf-8'))
    first=next(i for i,end in enumerate(all_ends) if end>header_bytes)
    starts=([0]+all_ends[:-1])[first:];ends=all_ends[first:]
    mapping=[]
    for _,_,_,c0,c1 in words:
        byte0=len(unicodedata.normalize('NFC',text[:len(head)+int(c0)]).encode('utf-8'))
        byte1=len(unicodedata.normalize('NFC',text[:len(head)+int(c1)]).encode('utf-8'))
        matches=[i for i in range(len(ends)) if starts[i]<byte1 and ends[i]>byte0]
        if not matches:raise ValueError('An aligned word has no lyric token.')
        mapping.append(matches)
    frames=len(item['codec'])
    selective=item.get('alignment_method')=='whisper'
    if selective:
        # Estimated lyric positions stay in the review JSON, but only observed
        # acoustic word intervals supervise the cursor.
        indices=np.full(frames,-1,dtype=np.int32)
        times=(np.arange(frames)+.5)/25
        for index,(start,end,weight,_,_) in enumerate(words):
            if weight>0 and end>start:
                indices[(times>=start)&(times<end)]=index
        active=np.flatnonzero(indices>=0)
        if not len(active):return None
        chosen=indices[active]
    else:
        indices=np.clip(np.searchsorted(words[:,0],np.arange(frames)/25,side='right')-1,0,len(words)-1)
        active=np.arange(frames);chosen=indices
    rows=[];cols=[];values=[]
    normalizer=0.
    for row,word in enumerate(chosen):
        tokens=mapping[word];weight=float(words[word,2]) if selective else 1.
        rows.extend([row]*len(tokens));cols.extend(tokens);values.extend([weight/len(tokens)]*len(tokens))
        normalizer+=weight
    base=(1+first,1+len(ids_full),frames,
          torch.tensor(rows,device=device),torch.tensor(cols,device=device),torch.tensor(values,device=device))
    return base+(torch.tensor(active,device=device),normalizer) if selective else base


def training_attention(q,k,v,query_chunk_size=256):
    """Exact causal attention with bounded math-SDPA forward AND backward.

    Windows builds may choose math SDPA for grouped-query attention. A whole
    9206-token, 16-head FP32 score matrix alone costs 5.05 GiB. Query tiling
    keeps every visible key; checkpointing each tile prevents autograd from
    retaining all tiles' quadratic intermediates until the layer backward.
    This is not a sliding window, song crop, or detached/truncated gradient.
    """
    if query_chunk_size<1 or len(q)!=len(k) or k.shape!=v.shape:
        raise ValueError('Invalid causal attention tiling inputs.')
    def tile(query,key,value,start):
        query=query.transpose(0,1)[None]
        key=key.transpose(0,1)[None];value=value.transpose(0,1)[None]
        mask=None
        if start:
            mask=torch.arange(key.shape[-2],device=q.device)[None,:]<=torch.arange(start,start+query.shape[-2],device=q.device)[:,None]
        result=F.scaled_dot_product_attention(query,key,value,attn_mask=mask,is_causal=start==0,
                                             enable_gqa=query.shape[1]!=key.shape[1])
        return result[0].transpose(0,1)
    output=[]
    for start in range(0,len(q),query_chunk_size):
        end=min(start+query_chunk_size,len(q))
        args=(q[start:end],k[:end],v[:end],start)
        if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
            output.append(checkpoint(tile,*args,use_reentrant=False))
        else:output.append(tile(*args))
    return torch.cat(output,dim=0)


def ar_layer(layer,x,cos,sin):
    q,k,v=layer.self_attn.project_qkv(layer.input_layernorm(x),cos,sin)
    h=training_attention(q[0],k[0],v[0])
    x=x+layer.self_attn.o_proj(h.flatten(1)[None])
    return x+layer.mlp(layer.post_attention_layernorm(x))


def loss(model,ids,prefix_length,cursor_head=None,cursor=None,grad=True,chunk=256):
    bb=model.model;x=bb.embed_tokens(ids)
    cos,sin=bb.rotary_emb(torch.arange(ids.shape[1],device=ids.device)[None])
    for layer in bb.layers:
        x=checkpoint(ar_layer,layer,x,cos,sin,use_reentrant=False) if grad else ar_layer(layer,x,cos,sin)
    hn=bb.norm(x[0]);h=hn[prefix_length-1:-1];targets=ids[0,prefix_length:]
    def ce(hidden,target):return F.cross_entropy(model.lm_head(hidden).float(),target,reduction='sum')
    total=0.
    for start in range(0,len(h),chunk):
        args=(h[start:start+chunk],targets[start:start+chunk])
        total=total+(checkpoint(ce,*args,use_reentrant=False) if grad else ce(*args))
    lm=total/len(h)
    if cursor is None:return lm,None
    j0,j1,frames,rows,cols,values=cursor[:6]
    if len(cursor)==8:
        active,normalizer=cursor[6:]
        q=cursor_head(h[active].float())
    else:
        normalizer=frames;q=cursor_head(h[:frames].float())
    keys=hn[j0:j1].float()
    logp=(q@keys.T/math.sqrt(q.shape[-1])).log_softmax(-1)
    return lm,-(logp[rows,cols]*values).sum()/normalizer

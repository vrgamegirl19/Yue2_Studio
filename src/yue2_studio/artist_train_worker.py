"""Separate-process Artist AR LoRA worker. Never runs on import."""
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def run(spec,directory):
    if spec.get('gpu_confirmed') is not True:raise ValueError('GPU preparation/training was not approved.')
    from .settings import ROOT
    os.environ['HF_MODULES_CACHE']=str(ROOT/'hf_modules_artist')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from torch import nn
    from .training_worker import check_cancel,Cancelled,model_identity
    from .jobs import write_json
    from .artist_training import validate_project
    from .artist_encoder import REVISION,HASHES,load_checkpoint
    from .artist_prepare import prepare,regularizers
    from .artist_ar import install,export,sequence,cursor_targets,loss,place_ar
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.tokenization_yue2 import YuE2TextTokenizer
    check_cancel(directory);validate_project(spec['project'])
    from .artist_setup import verify_asset
    paths={kind:Path(verify_asset(kind,folder)) for kind,folder in spec['paths'].items()}
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise ValueError('A BF16-capable NVIDIA GPU is required.')
    # Leave display/driver headroom on Windows; fail explicitly instead of
    # allocating up to physical capacity and risking slow WDDM memory paging.
    torch.cuda.set_per_process_memory_fraction(.88)
    result=directory/'result';result.mkdir(exist_ok=True)
    device=torch.device('cuda');controls=spec['controls'];project=spec['project']
    # Verify both pinned checkpoints before any expensive preparation.
    for name in HASHES:load_checkpoint(paths['encoder'],name)
    base_identity=model_identity(paths['model'],directory)
    tokenizer=YuE2TextTokenizer(paths['model']/'qwen.tiktoken')
    minted,mval=regularizers(tokenizer,paths['regularizer'])
    items=prepare(spec,directory,tokenizer,device)
    train=[x for x in items if x['name']!=spec['holdout']]
    holdout=[x for x in items if x['name']==spec['holdout']]
    if not train or len(holdout)!=1:raise ValueError('Invalid artist holdout split.')
    check_cancel(directory);gc.collect();torch.cuda.empty_cache()
    print('Loading frozen YuE2 base and installing AR-only trainable adapters.',flush=True)
    model=place_ar(YuE2ForCausalLM.from_pretrained(paths['model'],local_files_only=True,dtype=torch.bfloat16),device)
    print('Unused frozen NAR branch remains on CPU; CUDA allocator limited to 88% of device memory.',flush=True)
    torch.manual_seed(42);rng=random.Random(42)
    adapters=install(model,controls['rank'])
    cursor_head=nn.Linear(model.config.hidden_size,model.config.hidden_size,bias=False,device=device,dtype=torch.float32)
    nn.init.eye_(cursor_head.weight)
    parameters=[p for adapter in adapters.values() for p in (adapter.down,adapter.up)]
    if controls['alignment_weight']:parameters+=list(cursor_head.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=controls['learning_rate'],betas=(.9,.95),weight_decay=0.)
    metadata={'rank':str(controls['rank']),'trigger_word':project['trigger'],'training_style':project['shared_style'],'project_id':project['id'],
              'encoder_revision':REVISION,'companion_sha256':HASHES['nar_lora_joint_v4.pt'],
              'conditioning':'cot-off','targets':'ar-attention-and-mlp','seed':'42'}
    write_json(result/'manifest.json',dict(format='yue2-artist-ar-v1',base=base_identity,controls=controls,
               companion=dict(path=str(paths['encoder']/'nar_lora_joint_v4.pt'),sha256=HASHES['nar_lora_joint_v4.pt']),
               metadata=metadata,holdout=spec['holdout'],inference_integration='artist-bundle-v1',resume_supported=False,
               memory_policy='AR-only GPU; allocator cap 88%; exact 256-query attention tiles with per-tile gradient checkpointing'))
    cursors={}
    if controls['alignment_weight']:
        for item in train:cursors[item['name']]=cursor_targets(item,tokenizer,device)
        if controls.get('alignment_method')=='whisper':
            usable=sum(cursor is not None for cursor in cursors.values())
            print(f'Whisper timing supervision available for {usable}/{len(train)} training songs; '
                  'estimated lyric positions supervise no frames.',flush=True)
            if not usable:
                raise ValueError('Whisper recognized no usable lyric intervals in the training songs. '
                                 'Use alignment weight 0 or review the audio/lyrics.')
    def save(name,step):export(adapters,result/name,{**metadata,'step':str(step)})
    @torch.no_grad()
    def evaluate(step):
        values={}
        for tag,rows in [('artist_holdout',holdout),('reference_holdout',mval)]:
            scores=[]
            for item in rows:
                check_cancel(directory);ids,prefix=sequence(item,device)
                value=loss(model,ids,prefix,grad=False)[0]
                if not torch.isfinite(value):raise ValueError('Non-finite validation loss.')
                scores.append(value.item())
            values[tag]=sum(scores)/len(scores)
        with (result/'metrics.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(dict(step=step,**values))+'\n')
        print(f'Validation step {step}: '+json.dumps(values)+' (lower loss is not proof of voice similarity)',flush=True)
        return values['artist_holdout']
    last_step=0;last_loss=None;start=time.monotonic();steps=controls['steps'];update_in_progress=False
    try:
        print('[YuE2] Starting Artist validation: baseline before training',flush=True)
        best=evaluate(0)
        print('[YuE2] Completed Artist validation: baseline recorded',flush=True)
        print(f'[YuE2] Starting Artist LoRA training: 0/{steps} steps',flush=True)
        for step in range(1,steps+1):
            check_cancel(directory);optimizer.zero_grad(set_to_none=True)
            factor=min(1.,step/50)*(.2+.8*.5*(1+math.cos(math.pi*min(step,3000)/3000)))
            for group in optimizer.param_groups:group['lr']=controls['learning_rate']*factor
            combined=0.
            for _ in range(2):
                check_cancel(directory)
                item=rng.choice(train if rng.random()<.5 else minted)
                ids,prefix=sequence(item,device)
                print(f'Training example: step {step}, {item["name"]}, {ids.shape[1]} tokens',flush=True)
                lm,cl=loss(model,ids,prefix,cursor_head,cursors.get(item['name']) if item['src']=='artist' else None)
                objective=lm+(controls['alignment_weight']*cl if cl is not None else 0)
                if not torch.isfinite(objective):raise ValueError('Non-finite training loss. Earlier checkpoints are retained.')
                (objective/2).backward();combined+=objective.item()/2
                # Do not keep the last example's graph roots alive while building
                # the next full-song graph. Optimizer gradients still accumulate.
                del objective,lm,cl,ids
            torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            update_in_progress=True
            optimizer.step()
            last_step=step;last_loss=combined
            update_in_progress=False
            elapsed=time.monotonic()-start
            print(f'[YuE2] Running Artist LoRA training: {step}/{steps} steps · loss {combined:.5f} · elapsed {elapsed:.0f}s · GPU allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB / peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB',flush=True)
            if step%controls['checkpoint_every']==0:save(f'step-{step:06d}.safetensors',step)
            if step%100==0 or step==steps:
                score=evaluate(step)
                if score<best:best=score;save('best.safetensors',step)
                save('last.safetensors',step)
        check_cancel(directory);save('final.safetensors',last_step)
        write_json(result/'training.json',dict(status='complete',steps=last_step,loss=last_loss,elapsed_seconds=time.monotonic()-start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   adapter=str(result/'final.safetensors'),note='Actual AR artist LoRA. Playback requires the pinned NAR companion, No score, Torch, no quantization and AR offloading disabled. No automatic comparison song.'))
        print(f'[YuE2] Completed Artist LoRA training: {steps}/{steps} steps',flush=True)
    except Cancelled:
        if last_step:save(f'stopped-{last_step:06d}.safetensors',last_step)
        write_json(result/'training.json',dict(status='cancelled',steps=last_step,loss=last_loss))
        raise
    except torch.OutOfMemoryError:
        # Backward OOM leaves incomplete gradients, but parameters still represent
        # the last completed optimizer update. Preserve an inference snapshot;
        # never advertise it as a resumable optimizer checkpoint or completed run.
        optimizer.zero_grad(set_to_none=True)
        gc.collect();torch.cuda.empty_cache()
        if last_step and not update_in_progress:save(f'oom-stopped-{last_step:06d}.safetensors',last_step)
        write_json(result/'training.json',dict(status='failed',reason='cuda_oom',steps=last_step,
                   loss=last_loss,resume_supported=False,
                   note=('OOM inside optimizer update; only earlier committed checkpoints are valid.' if update_in_progress else
                         'Full-song training stopped; last completed adapter snapshot retained if at least one step finished.')))
        raise


def main():
    path=Path(sys.argv[1]).resolve()
    from .training_worker import Cancelled
    try:run(json.loads(path.read_text(encoding='utf-8')),path.parent)
    except Cancelled:print('[YuE2] Cancelled Artist training: completed checkpoints retained',flush=True)


if __name__=='__main__':main()

"""Whole-song semantic encoding and optional English lyric alignment.

Only called by the explicitly approved queue worker. Original files are read-only.
"""
import gc
import hashlib
import json
import re
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file,save_file
from .settings import ROOT
from .artist_encoder import load_head,predict_tokens,sha256,REVISION
from .artist_training import validate_project,REG_SHA256
from .training_worker import check_cancel,model_identity
from .jobs import write_json


def words_of(lyrics):
    words=[];offset=0
    for line in lyrics.split('\n'):
        if not re.fullmatch(r'\s*\[.*\]\s*',line):
            for match in re.finditer(r'\S+',line):
                raw=match.group().lower().replace('’',"'")
                if any(c.isalpha() and c not in 'abcdefghijklmnopqrstuvwxyz' for c in raw):
                    raise ValueError('Lyric alignment currently supports English letters only. Use alignment weight 0 for other languages.')
                word=re.sub("[^a-z']",'',raw)
                if re.search('[a-z]',word):words.append((offset+match.start(),offset+match.end(),word))
        offset+=len(line)+1
    if not words:raise ValueError('No alignable lyric words.')
    return words


def read_audio(path,rate,stereo=False):
    from scipy.signal import resample_poly
    audio,sr=sf.read(path,dtype='float32',always_2d=True)
    if audio.shape[1] not in (1,2) or not len(audio) or not np.isfinite(audio).all():raise ValueError('Expected finite mono/stereo audio.')
    if stereo:
        if audio.shape[1]==1:audio=np.repeat(audio,2,axis=1)
    else:audio=audio.mean(1)
    g=gcd(sr,rate)
    return resample_poly(audio,rate//g,sr//g,axis=0).astype(np.float32)


def regularizers(tokenizer,folder):
    from yue2.protocol import SongRequest,token_prefixes
    path=Path(folder)/'regularizer/minted_regularizer_pack.pt'
    if sha256(path)!=REG_SHA256:raise ValueError('Reference pack hash mismatch.')
    # Pinned data-only NumPy globals; never unrestricted unpickling.
    with torch.serialization.safe_globals([np._core.multiarray._reconstruct,np.ndarray,np.dtype,np.dtypes.Int32DType]):
        pack=torch.load(path,map_location='cpu',weights_only=True)
    train=[];validation=[];excluded=0
    for row in pack:
        if row['src'] not in ('minted','minted_val'):raise ValueError('Unexpected reference split.')
        codec=row['codec']
        if not isinstance(codec,np.ndarray) or codec.ndim!=1 or codec.dtype!=np.int32 or not len(codec) or codec.min()<0 or codec.max()>=32768:
            raise ValueError('Invalid reference codec.')
        item=dict(row,prefix=token_prefixes(SongRequest(style=row['style'],lyrics=row['lyrics'],cot='off'),tokenizer))
        if len(item['prefix'])+len(codec)+1>12288:excluded+=1;continue
        (train if row['src']=='minted' else validation).append(item)
    if not train or not validation:raise ValueError('Reference train/validation split empty.')
    print(f'Reference pack: {len(train)} training, {len(validation)} validation, {excluded} oversized songs excluded (not clipped).',flush=True)
    return train,validation[:6]


def prepare(spec,directory,tokenizer,device):
    from transformers import AutoModel,AutoFeatureExtractor
    from yue2.protocol import SongRequest,token_prefixes
    paths={k:Path(v) for k,v in spec['paths'].items()}
    project=spec['project'];rows=validate_project(project);aligned=spec['controls']['alignment_weight']>0
    method=spec['controls'].get('alignment_method','mms')
    if method not in ('mms','whisper'):raise ValueError('Unknown artist lyric timing method.')
    identities=model_identity(paths['mert'],directory)
    cache=ROOT/'training/artist-cache';cache.mkdir(parents=True,exist_ok=True)
    items=[]
    for row in rows:
        check_cancel(directory)
        if aligned:words_of(row['lyrics'])
        source=Path(project['folder'])/row['name'];audio_hash=sha256(source)
        identity=dict(version='artist-whole-song-v1',audio=audio_hash,lyrics=row['lyrics_sha256'],mert=identities,encoder=REVISION,
                      alignment=('htdemucs-ft-default-stable-ts-2.19.1-large-v3-reference-scene-words-v1' if method=='whisper'
                                 else 'htdemucs-shifts0-mms-fa-english-v1') if aligned else 'none')
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        folder=cache/key;folder.mkdir(exist_ok=True)
        style=project['trigger']+', '+project['shared_style']
        item=dict(name=row['name'],src='artist',style=style,lyrics=row['lyrics'],source=source,folder=folder,
                  identity=identity,alignment_method=method if aligned else 'none')
        item['prefix']=token_prefixes(SongRequest(style=style,lyrics=row['lyrics'],cot='off'),tokenizer)
        if len(item['prefix'])+round(row['seconds']*25)+1>12288:raise ValueError('Full song plus lyrics exceeds context: '+row['name'])
        try:
            receipt=json.loads((folder/'ready.json').read_text())
            if receipt['identity']!=identity or receipt['sha256']!=sha256(folder/'data.safetensors'):raise ValueError('Stale cache')
            data=load_file(str(folder/'data.safetensors'))
            cached={'codec':data['codec'].numpy()}
            if aligned:cached['words']=data['words'].numpy()
            if aligned and method=='whisper':
                review=json.loads((folder/'alignment.json').read_text(encoding='utf-8'))
                if (not isinstance(review,dict) or review.get('method')!='whisper-assisted' or review.get('song')!=row['name']
                    or review.get('source_audio_sha256')!=audio_hash
                    or review.get('lyrics_sha256')!=row['lyrics_sha256']
                    or len(review.get('words',[]))!=len(words_of(row['lyrics']))):
                    raise ValueError('Stale Whisper timing review')
                cached['alignment_review']=review
            item.update(cached)
        except (OSError,ValueError,KeyError):pass
        items.append(item)
    missing=[x for x in items if 'codec' not in x]
    print(f'[YuE2] Starting Artist encoding: {len(items)-len(missing)}/{len(items)} items',flush=True)
    if missing:
        mert_path=paths['mert']
        processor=AutoFeatureExtractor.from_pretrained(mert_path,local_files_only=True)
        mert=AutoModel.from_pretrained(mert_path,local_files_only=True,trust_remote_code=True).eval().to(device)
        head=load_head(paths['encoder']).to(device)
        for index,item in enumerate(missing):
            check_cancel(directory);mono=read_audio(item['source'],24000);chunks=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for start in range(0,len(mono),720000):
                    check_cancel(directory);segment=mono[start:start+720000]
                    if len(segment)<24000:break
                    inputs=processor([segment],sampling_rate=24000,return_tensors='pt')
                    output=mert(**{k:v.to(device) for k,v in inputs.items()},output_hidden_states=True)
                    chunks.append(output.hidden_states[20].reshape(-1,1024).float().cpu())
                    del output,inputs
            features=torch.cat(chunks)
            features=torch.nn.functional.interpolate(features.T[None],size=round(len(mono)/24000*25),mode='linear',align_corners=False)[0].T.numpy().astype(np.float16)
            item['codec']=predict_tokens(head,features,device)
            print(f'[YuE2] Running Artist encoding: {len(items)-len(missing)+index+1}/{len(items)} items · {item["name"]}',flush=True)
        del mert,head,features,chunks;gc.collect();torch.cuda.empty_cache()
    print(f'[YuE2] Completed Artist encoding: {len(items)}/{len(items)} items',flush=True)
    if aligned and missing:
        import torchaudio
        from scipy.signal import resample_poly
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        torch.hub.set_dir(str(ROOT/'models/artist-tools'))
        separator='htdemucs_ft' if method=='whisper' else 'htdemucs'
        print(f'Loading {separator} vocal separator (first use downloads its official weights).',flush=True)
        demucs=get_model(separator).eval()
        print(f'[YuE2] Starting Vocal separation: 0/{len(missing)} items',flush=True)
        for index,item in enumerate(missing):
            check_cancel(directory)
            audio=torch.from_numpy(read_audio(item['source'],demucs.samplerate,stereo=True).T.copy())
            with torch.inference_mode():
                if method=='whisper':
                    stems=apply_model(demucs,audio[None],device=device,progress=False)[0]
                else:
                    ref=audio.mean(0);mean=ref.mean();std=ref.std()
                    if std<1e-7:raise ValueError('Silent recording cannot be aligned.')
                    stems=apply_model(demucs,((audio-mean)/std)[None],device=device,shifts=0,split=True,progress=False)[0]*std+mean
            vocals=stems[demucs.sources.index('vocals')].mean(0).cpu().numpy()
            g=gcd(demucs.samplerate,16000)
            sf.write(item['folder']/'vocals.wav',resample_poly(vocals,16000//g,demucs.samplerate//g),16000,subtype='FLOAT')
            print(f'[YuE2] Running Vocal separation: {index+1}/{len(missing)} items',flush=True)
        print(f'[YuE2] Completed Vocal separation: {len(missing)}/{len(missing)} items',flush=True)
        del demucs,stems,audio;gc.collect();torch.cuda.empty_cache()
        if method=='whisper':
            try:import stable_whisper
            except ImportError as exc:raise ValueError('Whisper-assisted timing requires stable-ts in .venv-artist.') from exc
            from .artist_whisper import match_lyrics
            print('Loading stable-ts Whisper large-v3 for acoustic lyric timing.',flush=True)
            whisper=stable_whisper.load_model('large-v3',device=device)
            print(f'[YuE2] Starting Whisper lyric timing: 0/{len(missing)} items',flush=True)
            for index,item in enumerate(missing):
                check_cancel(directory)
                stem=item['folder']/'vocals.wav';audio=read_audio(stem,16000)
                result=whisper.transcribe(str(stem),language='en',word_timestamps=True,verbose=False)
                check_cancel(directory)
                timed,summary=match_lyrics(item['lyrics'],result,len(audio)/16000)
                item['words']=np.asarray([[w['start'],w['end'],w['training_weight'],w['start_offset'],w['end_offset']]
                                          for w in timed],dtype=np.float32)
                item['alignment_review']=dict(version=1,method='whisper-assisted',song=item['name'],
                                               source_audio_sha256=item['identity']['audio'],
                                               lyrics_sha256=item['identity']['lyrics'],summary=summary,words=timed)
                print(f'[YuE2] Running Whisper lyric timing: {index+1}/{len(missing)} items · '
                      f'{summary["counts"]["exact"]} exact, {summary["counts"]["fuzzy"]} fuzzy, '
                      f'{summary["counts"]["estimated"]} estimated (estimates excluded from timing loss)',flush=True)
                del result,audio
            print(f'[YuE2] Completed Whisper lyric timing: {len(missing)}/{len(missing)} items',flush=True)
            del whisper;gc.collect();torch.cuda.empty_cache()
        else:
            bundle=torchaudio.pipelines.MMS_FA
            print('Loading MMS forced aligner (first use downloads its official weights).',flush=True)
            aligner=bundle.get_model(with_star=False,dl_kwargs={'model_dir':str(ROOT/'models/artist-tools/checkpoints')}).eval().to(device)
            labels={c:i for i,c in enumerate(bundle.get_labels(star=None))}
            print(f'[YuE2] Starting Lyric alignment: 0/{len(missing)} items',flush=True)
            for index,item in enumerate(missing):
                check_cancel(directory);words=words_of(item['lyrics']);tokens=[[labels[c] for c in word] for _,_,word in words]
                audio=read_audio(item['folder']/'vocals.wav',16000)
                with torch.inference_mode():em=aligner(torch.from_numpy(audio[None]).to(device))[0].log_softmax(-1).cpu()
                check_cancel(directory)
                alignment,scores=torchaudio.functional.forced_align(em,torch.tensor([[c for word in tokens for c in word]],dtype=torch.int64),blank=0)
                spans=torchaudio.functional.merge_tokens(alignment[0],scores[0].exp());scale=len(audio)/16000/em.shape[1]
                result=[];offset=0
                for word,tok in zip(words,tokens):
                    segment=spans[offset:offset+len(tok)];offset+=len(tok)
                    if len(segment)!=len(tok):raise ValueError('Incomplete lyric alignment.')
                    result.append([segment[0].start*scale,segment[-1].end*scale,float(np.mean([s.score for s in segment])),word[0],word[1]])
                item['words']=np.asarray(result,dtype=np.float32)
                print(f'[YuE2] Running Lyric alignment: {index+1}/{len(missing)} items · mean confidence {item["words"][:,2].mean():.3f} (not a quality guarantee)',flush=True)
            print(f'[YuE2] Completed Lyric alignment: {len(missing)}/{len(missing)} items',flush=True)
            del aligner,em;gc.collect();torch.cuda.empty_cache()
    for item in items:
        check_cancel(directory)
        if sha256(item['source'])!=item['identity']['audio']:raise ValueError('Audio changed during preparation.')
    validate_project(project)
    for item in missing:
        data={'codec':torch.from_numpy(item['codec'].copy())}
        if aligned:data['words']=torch.from_numpy(item['words'].copy())
        target=item['folder']/'data.safetensors';temporary=target.with_suffix('.tmp')
        save_file(data,str(temporary));temporary.replace(target)
        if aligned and method=='whisper':write_json(item['folder']/'alignment.json',item['alignment_review'])
        write_json(item['folder']/'ready.json',dict(identity=item['identity'],sha256=sha256(target)))
    if aligned and method=='whisper':
        review_dir=directory/'result/alignment';review_dir.mkdir(parents=True,exist_ok=True)
        overview=[]
        for item in items:
            check_cancel(directory)
            filename=hashlib.sha256(item['name'].encode('utf-8')).hexdigest()[:12]+'.json'
            write_json(review_dir/filename,item['alignment_review'])
            overview.append(dict(song=item['name'],file=filename,summary=item['alignment_review']['summary']))
        write_json(review_dir/'index.json',dict(version=1,method='whisper-assisted',songs=overview))
    write_json(directory/'result/dataset.json',dict(songs=[dict(name=x['name'],cache=str(x['folder']),identity=x['identity']) for x in items],holdout=spec['holdout']))
    print(f'[YuE2] Completed Artist preparation: {len(items)}/{len(items)} items',flush=True)
    return items

"""Explicit, CPU-only Artist setup. Importing this module never downloads or installs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from .settings import ROOT

# Pinned files from the locally tested releases. Model code is not executed here.
ASSETS = {
    'model':dict(repo='m-a-p/YuE2-3B',revision='1a96eca688d6ae5d7f0feb88573fec89920fcd19',files={
        'model.safetensors':'1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9'},support=['config.json','qwen.tiktoken','generation_config.json','yue2_generation_config.json']),
    'vae':dict(repo='m-a-p/YuE2-Vae',revision='95535e72a97bc0f09b8ada125d26b4009428c0e8',files={
        'model.safetensors':'807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346'},support=['config.json']),
    'mert':dict(repo='m-a-p/MERT-v2-FullSong',revision='d8ba1c745e733b3908ce6ad16ebeb17ac7600a42',files={
        'model.safetensors':'e6dd2ab187d6dd62b6521cd7d8f932e237acf0c5757745a7232082e28391350d',
        'configuration_mert2.py':'77b53ec9d7ee31a599d744fb006e812c7eeaf7390deb46e2f460cf8c17b00bd6',
        'modeling_mert2.py':'b1a3174e5649c4b26b0c90d8626f0adacfbbba111a58ed3bb72ad651945a2f5c'},support=['config.json','preprocessor_config.json']),
    'encoder':dict(repo='Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4',revision='f2278a2e005dc4ecc421c53a0929f62b3aeb2280',files={
        'tokenizer_head_joint_v4.pt':'d23c4f757a05f031134b8471ec84245ec2338966516e1a9e26a17ff300a5f87e',
        'nar_lora_joint_v4.pt':'df175dbf9405a8e15b2c3f8dbdcc97303575f763787f227b03029020e28102fe'},support=[]),
    'regularizer':dict(repo='Mothersuperior/yue2-minted-corpus',repo_type='dataset',revision='5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35',files={
        'regularizer/minted_regularizer_pack.pt':'bdd9b9780de46bb0752c3e3bc101869759493443c6e35b1b2d4a2c5033eadc4e'},support=[]),
}
REQUIRED = {'model':['config.json','qwen.tiktoken'],'vae':['config.json'],
            'mert':['config.json','preprocessor_config.json'],'encoder':[],'regularizer':[]}
PACKAGES = ['transformers==4.57.6','numpy==2.2.6','soundfile==0.13.1','safetensors==0.7.0',
            'huggingface-hub==0.36.2','accelerate==1.13.0','tiktoken==0.12.0','scipy==1.15.3',
            'dora-search==0.1.12','julius==0.2.7','lameenc==1.8.1','openunmix==1.3.0',
            'omegaconf==2.3.0','antlr4-python3-runtime==4.9.3','submitit==1.5.3',
            'treetable==0.2.6','retrying==1.4.2','cloudpickle==3.1.1','einops==0.8.1',
            'stable-ts==2.19.1','openai-whisper==20250625','more-itertools==10.8.0',
            'numba==0.61.2','llvmlite==0.44.0']
PROBE = """import json,sys,importlib.metadata as m
import torch,torchaudio,transformers,scipy,soundfile,safetensors,demucs,demucs.pretrained
assert sys.version_info[:2]==(3,12), 'Artist runtime requires Python 3.12'
assert torch.__version__=='2.10.0+cu130', 'Expected tested Torch 2.10.0+cu130'
assert torchaudio.__version__==torch.__version__, 'Torch/torchaudio mismatch'
assert transformers.__version__=='4.57.6', 'Expected tested Transformers 4.57.6'
assert not torch.cuda.is_initialized(), 'Setup must not initialize CUDA'
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'torchaudio':torchaudio.__version__,'cuda_initialized':False,'demucs':m.version('demucs')}))
"""
WHISPER_PROBE = """import json,importlib.metadata as m,shutil,torch,stable_whisper,whisper
assert m.version('stable-ts')=='2.19.1', 'Expected tested stable-ts 2.19.1'
assert m.version('openai-whisper')=='20250625', 'Expected tested openai-whisper 20250625'
assert shutil.which('ffmpeg'), 'Whisper timing requires the FFmpeg CLI in PATH'
assert not torch.cuda.is_initialized(), 'Whisper setup must not initialize CUDA'
print(json.dumps({'stable_ts':m.version('stable-ts'),'openai_whisper':m.version('openai-whisper'),'ffmpeg':True,'cuda_initialized':False}))
"""


def cpu_env():
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
    return env


def run_command(args, *, offline=True):
    env=cpu_env()
    # Ignore unrelated global pip indexes/trusted-host settings; keep TLS checks.
    for key in ('PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PIP_TRUSTED_HOST'):
        env.pop(key,None)
    env['PIP_CONFIG_FILE']=os.devnull
    env['PIP_USE_FEATURE']='truststore'  # Also inherited by pip build subprocesses.
    args=[str(a) for a in args]
    if args[1:3]==['-m','pip']:args.insert(3,'--use-feature=truststore')
    if not offline:
        env.pop('HF_HUB_OFFLINE',None);env.pop('TRANSFORMERS_OFFLINE',None)
    try:
        return subprocess.run(args,check=True,env=env,capture_output=True,text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except subprocess.CalledProcessError as exc:
        print((exc.stdout or '')[-12000:],flush=True)
        print((exc.stderr or '')[-12000:],file=sys.stderr,flush=True)
        raise


def write_record(path,data):
    from yue2.storage import write_json
    write_json(path,data)


def verify_asset(kind,folder):
    folder=Path(folder).resolve()
    for name in REQUIRED[kind]:
        if not (folder/name).is_file():raise ValueError(f'{kind}: missing {name}')
    if kind in ('model','vae'):
        from .trainer_setup import validate_model_folder
        validate_model_folder(str(folder),kind)
    if kind=='mert':
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
        if config.get('model_type')!='mert2':raise ValueError('Expected MERT2 config.')
    for name,expected in ASSETS[kind]['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder):raise ValueError('Asset path escapes its folder.')
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'{kind}: hash mismatch for {name}; existing file was not modified.')
    return str(folder)


def model_paths(overrides=None):
    overrides=overrides or {}
    if set(overrides)-set(ASSETS):raise ValueError('Unknown Artist model path.')
    old={}
    try:old=json.loads((ROOT/'training/artist-models.json').read_text(encoding='utf-8'))['paths']
    except (OSError,ValueError,KeyError,TypeError):pass
    defaults={'model':'YuE2-3B','vae':'YuE2-Vae','mert':'MERT-v2-FullSong',
              'encoder':'artist-encoder-v4','regularizer':'artist-regularizer'}
    # The existing Style downloader may already provide the full model/VAE.
    try:
        shared=json.loads((ROOT/'training/models.json').read_text(encoding='utf-8'))
        old={**{k:shared[k] for k in ('model','vae') if k in shared},**old}
    except (OSError,ValueError,TypeError):pass
    return {key:str(overrides.get(key) or old.get(key) or ROOT/'models'/name) for key,name in defaults.items()}


def download_models(*, confirmed=False, terms_accepted=False, paths=None):
    if confirmed is not True or terms_accepted is not True:
        raise ValueError('Confirm downloads and review/accept the model terms first.')
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    resolved={}
    for kind,folder in model_paths(paths).items():
        if Path(folder).exists():
            # Explicit/local folders are never repaired or overwritten automatically.
            resolved[kind]=verify_asset(kind,folder)
            continue
        asset=ASSETS[kind]
        target=ROOT/'models/artist-cache'/kind/asset['revision']
        print(f'Artist setup: downloading {kind} at {asset["revision"]}',flush=True)
        required=set(asset['files'])|set(REQUIRED[kind])
        names=required|set(asset['support'])|{'README.md','LICENSE','THIRD_PARTY_NOTICES.md'}
        for name in sorted(names):
            try:
                hf_hub_download(asset['repo'],filename=name,revision=asset['revision'],
                    repo_type=asset.get('repo_type','model'),token=False,local_dir=str(target))
            except EntryNotFoundError:
                if name in required:raise
        resolved[kind]=verify_asset(kind,target)
    record={'paths':resolved,'revisions':{k:v['revision'] for k,v in ASSETS.items()}}
    write_record(ROOT/'training/artist-models.json',record)
    return record


def install_runtime(*, confirmed=False):
    if confirmed is not True:raise ValueError('Confirm the separate Artist dependency installation first.')
    if sys.version_info[:2]!=(3,12):raise ValueError('Run setup with the Studio Python 3.12 environment.')
    # Fresh target each attempt: never mutate the working Studio or an old Artist env.
    target=ROOT/'.artist-runtimes'/uuid.uuid4().hex
    target.parent.mkdir(parents=True,exist_ok=True)
    run_command([sys.executable,'-m','venv',target])
    python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    print('Installing separate Artist dependencies; the base Studio environment is unchanged.',flush=True)
    run_command([python,'-m','pip','install','torch==2.10.0','torchaudio==2.10.0',
                 '--index-url','https://download.pytorch.org/whl/cu130'],offline=False)
    run_command([python,'-m','pip','install',*PACKAGES],offline=False)
    # Install the known dependency list above, then avoid re-resolving the
    # explicitly tested Torch/torchaudio pair through Demucs' loose requirements.
    run_command([python,'-m','pip','install','--no-deps','demucs==4.0.1'],offline=False)
    run_command([python,'-m','pip','check'])
    # Share only installed source code, not the live environment's site-packages.
    run_command([python,'-I','-c',
        "import pathlib,sys,sysconfig; pathlib.Path(sysconfig.get_path('purelib'),'yue2-studio-source.pth').write_text(sys.argv[1]+'\\n',encoding='utf-8')",str(ROOT/'src')])
    report=json.loads(run_command([python,'-I','-c',PROBE]).stdout.strip().splitlines()[-1])
    record={'python':str(python.resolve()),'probe':report,'note':'CPU imports only; GPU and alignment not validated.'}
    write_record(ROOT/'training/artist-runtime.json',record)
    return record


def check_setup(paths=None):
    issues=[];verified={}
    for kind,folder in model_paths(paths).items():
        try:verified[kind]=verify_asset(kind,folder)
        except (OSError,ValueError,TypeError) as exc:issues.append(f'{kind}: {exc}')
    runtime={}
    try:
        runtime=json.loads((ROOT/'training/artist-runtime.json').read_text(encoding='utf-8'))
        runtime['probe']=json.loads(run_command([runtime['python'],'-I','-c',PROBE]).stdout.strip().splitlines()[-1])
    except (OSError,ValueError,KeyError,IndexError,subprocess.SubprocessError) as exc:
        issues.append('Artist runtime is missing or incompatible: '+str(exc))
    return {'files_and_imports_ready':not issues,'issues':issues,'paths':verified,'runtime':runtime,
            'note':'Read-only CPU verification. Does not establish GPU memory, alignment readiness, or musical quality.'}


def check_whisper_runtime(python):
    """Check optional timing imports without touching CUDA or downloading weights."""
    try:
        return json.loads(run_command([python,'-I','-c',WHISPER_PROBE]).stdout.strip().splitlines()[-1])
    except (OSError,ValueError,IndexError,subprocess.SubprocessError) as exc:
        raise ValueError('Whisper-assisted timing needs stable-ts/openai-whisper in the separate Artist runtime '
                         'and FFmpeg on PATH. Use Install separate Artist runtime or the manual steps in '
                         'docs/artist-trainer.md. '
                         'The regular MMS method remains available.') from exc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','download-models','install-runtime'])
    parser.add_argument('--confirm',action='store_true')
    parser.add_argument('--accept-model-terms',action='store_true')
    for kind in ASSETS:parser.add_argument('--'+kind)
    args=parser.parse_args()
    paths={kind:getattr(args,kind) for kind in ASSETS if getattr(args,kind)}
    try:
        if args.action=='check':result=check_setup(paths)
        elif args.action=='download-models':result=download_models(confirmed=args.confirm,terms_accepted=args.accept_model_terms,paths=paths)
        else:result=install_runtime(confirmed=args.confirm)
        print(json.dumps(result,indent=2))
        if args.action=='check' and not result['files_and_imports_ready']:raise SystemExit(1)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        print('Artist setup failed: '+str(exc),file=sys.stderr);raise SystemExit(1)


if __name__=='__main__':main()

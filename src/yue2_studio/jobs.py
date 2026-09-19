"""Durable single-worker queue shared by generation and transcription."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import uuid

from .settings import ROOT, validate_settings

SCRIPTS = ROOT / 'skills/yue2-music/scripts'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))
from abc_tools import parse_abc, report, strip_chords


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path,data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def terminate_worker(process):
    # Windows venv python.exe is a redirector with a child interpreter. Killing only
    # the redirector leaves that interpreter (and CUDA memory) running.
    if os.name == 'nt' and hasattr(process,'pid'):
        result = subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode and process.poll() is None:
            raise RuntimeError('Could not stop the worker process tree: '+result.stderr.decode('utf-8','replace'))
    else:
        process.terminate()


def failure_summary(log, code):
    if 'USE_FLASH_ATTENTION was not enabled for build' in log:
        return 'This PyTorch build lacks Flash Attention for the optimized torch backend. The compatibility fix now keeps torch CUDA graphs with cuDNN/SDPA attention. Retry the saved song; your lyrics and style are saved.'
    lines=[line.strip() for line in log.splitlines() if line.strip()]
    errors=[line for line in lines if re.match(r'^(?:[\w.]+(?:Error|Exception)|RuntimeError):',line)]
    return errors[-1][:1200] if errors else f'Engine exited with code {code}. Open the run log for details.'


def is_cuda_oom(log):
    """Recognize Torch CUDA allocation failures without retrying unrelated errors."""
    text = str(log).casefold()
    return ('cuda out of memory' in text or 'torch.outofmemoryerror' in text or
            ('cuda' in text and 'out of memory' in text))


def oom_retry_spec(spec):
    """Keep the song identical while selecting the safest compatible memory path."""
    retry = deepcopy(spec)
    runtime = retry['settings']['runtime']
    artist = retry.get('lora',{}).get('kind') == 'artist'
    can_offload = runtime.get('backend') in ('torch','torch-eager') and not artist
    if can_offload:
        runtime['offload_ar'] = True
    return retry, {'reason':'cuda_oom','attempt':2,'offload_ar':bool(runtime.get('offload_ar')),
                   'artist_lora':artist,'allocator':'expandable_segments'}


def score_check(text, strip=False, keep_voice='both'):
    if not isinstance(text,str) or len(text)>500000:
        raise ValueError('ABC must be text under 500 KB.')
    if keep_voice not in ('both','Vocal','Ins','convert_vocal_to_ins'):
        raise ValueError('Select both voices, Vocal, Ins or convert_vocal_to_ins.')
    prepared = strip_chords(text,keep_voice) if strip else text
    return {'abc':prepared,'report':report(parse_abc(prepared))}


def generation_spec(payload):
    if set(payload)-{'title','mode','stage','request','settings','source_job','lora'}:
        raise ValueError('Unknown generation fields.')
    settings = validate_settings(payload.get('settings',{}))
    request = dict(payload.get('request',{}))
    allowed = {'style','lyrics','cot','seed','abc','cfg_scale','id'}
    if set(request)-allowed:
        raise ValueError('Unknown song request fields.')
    if 'cfg_scale' in request and request['cfg_scale'] != settings['generation']['cfg_scale']:
        raise ValueError('Guidance values disagree. Set CFG in advanced settings.')
    request['cfg_scale'] = settings['generation']['cfg_scale']
    request.setdefault('id','song')
    if isinstance(request.get('seed'),str) and re.fullmatch(r'[0-9]{1,19}',request['seed']):
        request['seed'] = int(request['seed'])
    from .loras import prepare
    lora = prepare(settings, request, payload.get('lora'))
    from yue2.protocol import SongRequest
    req = SongRequest(**request)
    if not req.style.strip():
        raise ValueError('Describe a musical style before generating.')
    if len(req.style)+len(req.lyrics)+len(req.abc or '')>500000:
        raise ValueError('Request text is too large.')
    mode = payload.get('mode','create')
    stage = payload.get('stage','audio')
    if mode not in ('create','cover') or stage not in ('audio','plan'):
        raise ValueError('Unsupported workflow or stage.')
    if settings['runtime']['backend']=='audio.cpp':
        from .gguf import validate
        validate(settings,stage)
    if stage=='plan' and req.cot=='off':
        raise ValueError('Direct audio has no symbolic plan. Choose Full or Melody.')
    if mode=='cover' and not req.abc:
        raise ValueError('A cover needs a reviewed ABC melody. Transcribe a recording or import a score first.')
    if req.abc:
        score = parse_abc(req.abc)
        if req.cot=='melody' and any(v.chords for v in score.voices.values()):
            raise ValueError('The score still contains chords. Use Prepare melody, or choose Full to retain harmony.')
    return {**({'lora':lora} if lora else {}), 'request':req.to_dict(),'settings':settings,'stage':stage,'mode':mode,
            'title':str(payload.get('title') or 'Untitled song')[:180], 'source_job':str(payload.get('source_job') or '')[:100]}


class JobManager:
    def __init__(self, root=None, start=True):
        self.root = Path(root or ROOT/'runs/studio').resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.uploads = self.root/'uploads'
        self.uploads.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        self.jobs = {}
        self.process = None
        self.active_id = None
        self.stopping = False
        for path in self.root.glob('*/job.json'):
            try:
                job = json.loads(path.read_text(encoding='utf-8'))
                job.setdefault('starred',False)
                if job.get('kind')=='generation' and 'backend' not in job:
                    try:
                        input_data = json.loads((path.parent/'input.json').read_text(encoding='utf-8'))
                        job['backend'] = input_data.get('settings',{}).get('runtime',{}).get('backend','torch')
                    except (OSError,ValueError,AttributeError):
                        job['backend'] = 'torch'
                if job['status'] in ('queued','running','cancelling'):
                    job.update(status='interrupted',error='Studio stopped before this job finished. Start a fresh run.',finished=now())
                    write_json(path,job)
                self.jobs[job['id']] = job
            except (ValueError,KeyError,OSError):
                continue
        self.thread = threading.Thread(target=self._loop,daemon=True,name='yue2-studio-queue')
        if start:
            self.thread.start()

    def directory(self,job_id):
        if not re.fullmatch(r'[a-f0-9]{32}',job_id) or job_id not in self.jobs:
            raise ValueError('Unknown run.')
        return self.root/job_id

    def _persist(self,job):
        write_json(self.directory(job['id'])/'job.json',job)

    def _add(self,kind,spec):
        with self.lock:
            if self.stopping:
                raise ValueError('Studio is shutting down.')
            job_id = uuid.uuid4().hex
            directory = self.root/job_id
            directory.mkdir(exist_ok=False)
            write_json(directory/'input.json',spec)
            job = dict(id=job_id,kind=kind,title=spec['title'],status='queued',created=now(),starred=False,stage=spec.get('stage','transcribe'),mode=spec.get('mode','cover'))
            if kind=='generation':
                job['backend'] = spec['settings']['runtime']['backend']
                if spec.get('lora'):
                    job['lora'] = deepcopy(spec['lora'])
            self.jobs[job_id] = job
            self._persist(job)
            self.queue.put(job_id)
            return deepcopy(job)

    def train(self,payload):
        from .trainer import training_spec
        return self._add('training',training_spec(payload))

    def train_artist(self,payload):
        from .artist_training import training_spec
        return self._add('artist_training',training_spec(payload))

    def prepare_artist(self,payload):
        from .artist_setup_worker import setup_spec
        spec=setup_spec(payload)
        with self.lock:
            if any(j['kind']=='artist_setup' and j['status'] in ('queued','running','cancelling') for j in self.jobs.values()):raise ValueError('An Artist setup action is already queued or running.')
            return self._add('artist_setup',spec)

    def download_training_models(self):
        with self.lock:
            if any(job['kind']=='trainer_setup' and job['status'] in ('queued','running','cancelling') for job in self.jobs.values()):
                raise ValueError('Training-model download is already queued or running.')
            return self._add('trainer_setup',dict(title='Download Style Trainer models',stage='setup',mode='trainer'))

    def generate(self,payload):
        return self._add('generation',generation_spec(payload))

    def transcribe(self,payload):
        if set(payload)-{'upload_id','settings','title'}:
            raise ValueError('Unknown transcription fields.')
        upload_id = str(payload.get('upload_id',''))
        if not re.fullmatch(r'[a-f0-9]{32}\.(wav|flac|mp3|ogg|m4a|aac|aiff|aif|opus)',upload_id) or not (self.uploads/upload_id).is_file():
            raise ValueError('Upload a source recording first.')
        settings = validate_settings(payload.get('settings',{}))
        executable = Path(settings['transcription']['python'])
        if not executable.is_file() or not re.fullmatch(r'python(?:\d+(?:\.\d+)*)?(?:\.exe)?',executable.name,re.I):
            raise ValueError('SheetSage2 Python executable was not found. Check Advanced → Cover transcription.')
        return self._add('transcription',{'upload_id':upload_id,'settings':settings,'title':str(payload.get('title') or 'Source transcription')[:180]})

    def list(self):
        with self.lock:
            return deepcopy(sorted(self.jobs.values(),key=lambda j:j['created'],reverse=True))

    def detail(self,job_id):
        with self.lock:
            directory = self.directory(job_id)
            job = deepcopy(self.jobs[job_id])
        log = directory/'run.log'
        if log.exists():
            with log.open('rb') as handle:
                handle.seek(max(0,log.stat().st_size-40000))
                job['log'] = handle.read().decode('utf-8','replace')
        else:
            job['log'] = 'Waiting for the GPU queue.'
        first_attempt = directory/'run.attempt-1.log'
        if first_attempt.is_file():
            job['attempt_logs'] = ['run.attempt-1.log']
        if job['status']=='failed' and job.get('failure_kind')!='cuda_oom':
            job['error']=failure_summary(job['log'],1)
        from .progress import read_progress
        job['progress']=read_progress(job['log'],job['status'])
        job['log_path']=str(log)
        adjustments=directory/'runtime_adjustments.json'
        if adjustments.exists():
            job['runtime_adjustments']=json.loads(adjustments.read_text(encoding='utf-8'))
        job['input'] = json.loads((directory/'input.json').read_text(encoding='utf-8'))
        if 'request' in job['input']:
            # Browser JSON numbers cannot retain all native 63-bit seed values.
            job['input']['request']['seed'] = str(job['input']['request']['seed'])
        result = directory/'result'
        job['artifacts'] = [str(p.relative_to(directory)).replace('\\','/') for p in sorted(result.rglob('*')) if p.is_file()] if result.exists() else []
        if (result/'score.abc').exists():
            job['abc'] = (result/'score.abc').read_text(encoding='utf-8')
        for name in ('result.json','studio_summary.json','transcription_manifest.json','training.json','training_models.json','lyrics-cleaning.json'):
            path = result/name
            if path.is_file():
                job.setdefault('receipts',{})[name] = json.loads(path.read_text(encoding='utf-8'))
        return job

    def cancel(self,job_id):
        with self.lock:
            self.directory(job_id)
            job = self.jobs[job_id]
            if job['status']=='queued':
                job.update(status='cancelled',finished=now())
            elif job['status']=='running':
                job['status']='cancelling'
                if self.active_id==job_id and self.process and self.process.poll() is None:
                    try:
                        if job['kind'] in ('training','artist_training'):
                            (self.directory(job_id)/'cancel.request').touch()
                        else:
                            terminate_worker(self.process)
                    except Exception:
                        job['status']='running'
                        raise
            self._persist(job)
            return deepcopy(job)

    def set_starred(self,job_id,starred):
        """Persist library metadata only; never alter a running worker or input."""
        if type(starred) is not bool:
            raise ValueError('Starred must be true or false.')
        with self.lock:
            self.directory(job_id)
            job = self.jobs[job_id]
            self._persist({**job,'starred':starred})
            job['starred'] = starred
            return deepcopy(job)

    def rename(self,job_id,title):
        """Change the library label without rewriting render inputs or artifacts."""
        if not isinstance(title,str) or not title.strip() or len(title.strip())>180:
            raise ValueError('Song name must contain 1 to 180 characters.')
        title = title.strip()
        with self.lock:
            self.directory(job_id)
            job = self.jobs[job_id]
            self._persist({**job,'title':title})
            job['title'] = title
            return deepcopy(job)

    def delete(self,job_id):
        with self.lock:
            directory = self.directory(job_id)
            job = self.jobs[job_id]
            if job['status'] in ('queued','running','cancelling'):
                raise ValueError('Cancel this run before removing it.')
            shutil.rmtree(directory)
            del self.jobs[job_id]
            return {'deleted':job_id,'title':job.get('title')}

    def _command(self,job_id,spec,input_name='input.json'):
        directory = self.directory(job_id)
        if self.jobs[job_id]['kind']=='artist_setup':
            return [sys.executable,'-u','-m','yue2_studio.artist_setup_worker',str(directory/'input.json')]
        if self.jobs[job_id]['kind']=='artist_training':
            return [spec['python'],'-u','-m','yue2_studio.artist_train_worker',str(directory/'input.json')]
        if self.jobs[job_id]['kind']=='trainer_setup':
            return [sys.executable,'-u','-m','yue2_studio.trainer_setup',str(directory/'input.json')]
        if self.jobs[job_id]['kind']=='training':
            return [sys.executable,'-u','-m','yue2_studio.training_worker',str(directory/'input.json')]
        if self.jobs[job_id]['kind']=='generation':
            return [sys.executable,'-u','-m','yue2_studio.worker',str(directory/input_name)]
        opts = dict(spec['settings']['transcription'])
        command = [opts.pop('python'),'-u',str(SCRIPTS/'transcribe.py'),str(self.uploads/spec['upload_id']),'--output',str(directory/'result')]
        for name,value in opts.items():
            if value is True:
                command.append('--'+name.replace('_','-'))
            elif value is not False and value is not None and value!='':
                command.extend(['--'+name.replace('_','-'),str(value)])
        return command

    def _loop(self):
        while True:
            job_id = self.queue.get()
            if job_id is None:
                self.queue.task_done()
                return
            try:
                with self.lock:
                    job = self.jobs[job_id]
                    if job['status']!='queued':
                        continue
                    directory = self.directory(job_id)
                    spec = json.loads((directory/'input.json').read_text(encoding='utf-8'))
                    command = self._command(job_id,spec)
                    env = os.environ.copy()
                    env.update(PYTHONPATH=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH',''),PYTHONUTF8='1',PYTHONUNBUFFERED='1')
                    env.setdefault('HF_HOME',str(ROOT/'hf-cache'))
                    env.setdefault('HF_MODULES_CACHE',str(ROOT/'hf_modules'))
                    self.active_id = job_id
                    job.update(status='running',started=now())
                    self._persist(job)
                    log = (directory/'run.log').open('w',encoding='utf-8')
                    try:
                        self.process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    except BaseException:
                        log.close()
                        raise
                try:
                    code = self.process.wait()
                finally:
                    log.close()
                # The failed worker has exited here, releasing its entire CUDA
                # context. Retry one time without changing song or quality inputs.
                if code and job['kind']=='generation' and job['status']=='running':
                    failure_path = directory/'run.log'
                    with failure_path.open('rb') as handle:
                        handle.seek(max(0,failure_path.stat().st_size-40000))
                        first_failure = handle.read().decode('utf-8','replace')
                    if is_cuda_oom(first_failure):
                        retry_spec, retry_info = oom_retry_spec(spec)
                        with self.lock:
                            if job['status']=='running':
                                failure_path.replace(directory/'run.attempt-1.log')
                                partial = directory/'result'
                                if partial.exists():
                                    shutil.rmtree(partial)
                                write_json(directory/'retry-input.json',retry_spec)
                                job['auto_retry'] = retry_info
                                self._persist(job)
                                retry_env = env.copy()
                                retry_env.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
                                retry_command = self._command(job_id,retry_spec,'retry-input.json')
                                retry_log = failure_path.open('w',encoding='utf-8')
                                try:
                                    self.process = subprocess.Popen(retry_command,cwd=ROOT,env=retry_env,
                                        stdout=retry_log,stderr=subprocess.STDOUT,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                                except BaseException:
                                    retry_log.close()
                                    raise
                            else:
                                retry_log = None
                        if retry_log is not None:
                            try:
                                code = self.process.wait()
                            finally:
                                retry_log.close()
                with self.lock:
                    if job['status']=='cancelling':
                        job['status']='cancelled'
                    elif code:
                        with (directory/'run.log').open('rb') as handle:
                            handle.seek(max(0,(directory/'run.log').stat().st_size-40000))
                            failure_log=handle.read().decode('utf-8','replace')
                        job.update(status='failed',error=failure_summary(failure_log,code))
                        if job.get('auto_retry'):
                            job['auto_retry']['exhausted'] = True
                            if is_cuda_oom(failure_log):
                                job['failure_kind'] = 'cuda_oom'
                                job['error'] = ('GPU memory ran out again after Yue2 automatically retried in a fresh process. '
                                    'The song may be too long for the available VRAM. Close other GPU tasks and try again.')
                    else:
                        job['status']='complete'
                        if job.get('auto_retry'):
                            job['auto_retry']['succeeded'] = True
                            job['warning'] = ('Recovered automatically after a CUDA out-of-memory error using a fresh GPU process'
                                + (' with AR offloading.' if job['auto_retry']['offload_ar'] else '.'))
                        manifest = directory/'result/transcription_manifest.json'
                        if manifest.exists():
                            warnings = json.loads(manifest.read_text(encoding='utf-8')).get('warnings',[])
                            if warnings:
                                job['status']='needs_review'
                                job['warning']='Transcription warnings: ' + '; '.join(map(str,warnings))
                        for name in ('result.json','studio_summary.json'):
                            path = directory/'result'/name
                            if path.exists():
                                receipt = json.loads(path.read_text(encoding='utf-8'))
                                if receipt.get('warnings'):
                                    job['status']='needs_review'
                                    job['warning']='; '.join(receipt['warnings'])
                                if any(receipt.get('truncated',{}).values()):
                                    job['status']='needs_review'
                                    job['warning']='A token limit was reached. Listen for an incomplete ending and inspect the score.'
                    job['finished']=now()
                    self._persist(job)
            except Exception as exc:
                with self.lock:
                    job = self.jobs[job_id]
                    job.update(status='failed',error=str(exc),finished=now())
                    self._persist(job)
            finally:
                with self.lock:
                    self.process = None
                    self.active_id = None
                self.queue.task_done()

    def close(self):
        with self.lock:
            self.stopping=True
            if self.active_id:
                self.cancel(self.active_id)
            for job in self.jobs.values():
                if job['status']=='queued':
                    job.update(status='interrupted',finished=now(),error='Studio closed before this run started.')
                    self._persist(job)
            self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=10)
        # Training cancels cooperatively, but shutdown must not orphan a GPU worker.
        with self.lock:
            if self.process and self.process.poll() is None:
                terminate_worker(self.process)

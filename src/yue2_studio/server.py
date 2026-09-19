"""Loopback-only local studio server. No web framework or build step required."""
from __future__ import annotations

import argparse
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import socket
import threading
import urllib.parse
import urllib.request
import uuid
import webbrowser

from . import llm
from .jobs import JobManager, score_check
from .settings import ROOT, GROUPS, FIXED, defaults, validate_settings
from .surprise import SurpriseManager
from .compatibility import capabilities
from .lifetime import BrowserLifetime
from .model_manager import ModelManager
from .dataset_import import DatasetImportManager

STATIC = Path(__file__).parent/'static'


def audius_config():
    """Return public browser OAuth configuration; never include the bearer token."""
    client_id = os.environ.get('YUE2_AUDIUS_API_KEY','').strip()
    return {'enabled':bool(client_id),'client_id':client_id}


def _unsupported_json(value):
    raise TypeError(f'Cannot encode {type(value).__name__}')


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        if os.name == 'nt':
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()

    def __init__(self,address,manager=None,root=None):
        self.browser_lifetime = BrowserLifetime()
        self.browser_stop = threading.Event()
        self.auto_stopping = False
        self.token = secrets.token_urlsafe(32)
        self.llm_lock = threading.Lock()
        super().__init__(address,Handler)
        self.jobs = manager or JobManager(root)
        self.surprises = SurpriseManager(self.jobs,self.llm_lock)
        self.models = ModelManager()
        self.datasets = DatasetImportManager()


    def service_actions(self):
        busy = (self.datasets.busy() or self.models.busy() or self.llm_lock.locked() or
                any(j['status'] in ('queued','running','cancelling') for j in self.jobs.list()) or
                any(b['status'] in ('queued','writing','rendering','cancelling') for b in self.surprises.list()))
        if not self.auto_stopping and self.browser_lifetime.should_stop(busy):
            self.auto_stopping = True
            print('Last Studio tab closed; no active work. Stopping server.',flush=True)
            threading.Thread(target=self.shutdown,daemon=True).start()

    def server_close(self):
        self.browser_stop.set()
        if hasattr(self,'datasets'):
            self.datasets.cancel()
        if hasattr(self,'models'):
            self.models.cancel()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'YuE2Studio/1.0'

    def log_message(self,*args):
        pass

    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self' https://api.audius.co https://*.audius.co; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        super().end_headers()

    def json(self,data,status=200):
        raw = json.dumps(data,ensure_ascii=False,default=lambda value: str(value) if isinstance(value,Fraction) else _unsupported_json(value)).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def guard(self,write=False):
        port = self.server.server_port
        hosts = {f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:
            raise PermissionError('Studio accepts loopback hosts only.')
        origin = self.headers.get('Origin')
        if origin and origin not in {'http://'+host for host in hosts}:
            raise PermissionError('Cross-origin requests are not allowed.')
        if write and not secrets.compare_digest(self.headers.get('X-Studio-Token',''),self.server.token):
            raise PermissionError('Studio session expired. Reload this page.')

    def body(self):
        if self.headers.get_content_type()!='application/json':
            raise ValueError('Expected JSON.')
        size = int(self.headers.get('Content-Length','0'))
        if not 0<size<=2*1024*1024:
            raise ValueError('JSON request must be under 2 MB.')
        data = json.loads(self.rfile.read(size))
        if not isinstance(data,dict):
            raise ValueError('Expected a JSON object.')
        return data

    def do_GET(self):
        try:
            self.guard()
            path = urllib.parse.urlsplit(self.path).path
            if path=='/api/browser-session':
                self.guard(write=True)
                self.browser_session()
            elif path=='/api/bootstrap':
                self.json({'token':self.server.token,'groups':GROUPS,'fixed':FIXED,'defaults':defaults(),
                    'providers':llm.catalogue(),'songwriter_prompt':llm.PROMPT,'compatibility':capabilities(),'profanity_check':True,'surprise_style_lock':True,'browser_autoclose':True,
                    # Audius documents its API key as a public OAuth client identifier. Never expose the bearer token here.
                    'audius':audius_config(),
                    'paths':{'root':str(ROOT),'runs':str(self.server.jobs.root)},
                    'installed':{name:(ROOT/'models'/name).is_dir() for name in ('YuE2-3B','YuE2-Vae','SheetSage2','MERT-v2-FullSong')}})
            elif path=='/api/dataset-import':
                self.json(self.server.datasets.status())
            elif path=='/api/models':
                self.json(self.server.models.status())
            elif path=='/api/loras':
                from .loras import catalogue
                self.json(catalogue())
            elif path=='/api/artist-trainer/projects':
                from .artist_trainer import projects
                self.json({'projects':projects()})
            elif path=='/api/artist-trainer/paths':
                from .artist_setup import model_paths
                self.json({'paths':model_paths()})
            elif re.fullmatch(r'/api/artist-trainer/projects/[a-f0-9]{32}',path):
                from .artist_trainer import load_project
                self.json(load_project(path.split('/')[-1]))
            elif path=='/api/trainer/projects':
                from .trainer import projects
                self.json({'projects':projects()})
            elif re.fullmatch(r'/api/trainer/projects/[a-f0-9]{32}',path):
                from .trainer import load_project
                self.json(load_project(path.split('/')[-1]))
            elif path=='/api/jobs':
                self.json({'jobs':self.server.jobs.list()})
            elif path=='/api/surprises':
                self.json({'batches':self.server.surprises.list()})
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}',path):
                self.json(self.server.jobs.detail(path.split('/')[-1]))
            elif path.startswith('/artifacts/'):
                parts = path.split('/',3)
                if len(parts)!=4:
                    raise ValueError('Invalid artifact URL.')
                directory = self.server.jobs.directory(parts[2])
                relative = urllib.parse.unquote(parts[3])
                file = (directory/relative).resolve()
                if not file.is_relative_to(directory.resolve()) or not file.is_file():
                    raise ValueError('Artifact not found.')
                self.file(file,download=not file.suffix.lower() in ('.flac','.wav','.mp3','.ogg'))
            elif path.startswith('/uploads/'):
                name = path.split('/')[-1]
                if not re.fullmatch(r'[a-f0-9]{32}\.[a-z0-9]+',name):
                    raise ValueError('Unknown upload.')
                self.file(self.server.jobs.uploads/name)
            elif path=='/dataset.js':
                self.file(STATIC/'dataset.js')
            elif path in ('/','/index.html','/app.js','/audius.js','/library.js','/models.js','/loras.js','/trainer.js','/artist.js','/style.css','/mark.svg'):
                self.file(STATIC/('index.html' if path=='/' else path[1:]))
            else:
                self.json({'error':'Not found.'},404)
        except (BrokenPipeError,ConnectionResetError):
            pass
        except PermissionError as exc:
            self.json({'error':str(exc)},403)
        except (ValueError,OSError,KeyError) as exc:
            self.json({'error':str(exc)},400)

    def browser_session(self):
        client = uuid.uuid4().hex
        self.server.browser_lifetime.opened(client)
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header('Content-Type','text/event-stream')
            self.send_header('Connection','close')
            self.end_headers()
            while not self.server.browser_stop.is_set():
                self.wfile.write(b': studio-connected\n\n')
                self.wfile.flush()
                if self.server.browser_stop.wait(2):
                    break
        except OSError:
            pass
        finally:
            self.server.browser_lifetime.closed(client)

    def do_POST(self):
        try:
            self.guard(write=True)
            path = urllib.parse.urlsplit(self.path).path
            if path=='/api/upload':
                self.upload()
                return
            data = self.body()
            if path=='/api/shutdown':
                self.json({'status':'stopping'})
                threading.Thread(target=self.server.shutdown,daemon=True).start()
            elif path=='/api/dataset-import/start':
                self.json(self.server.datasets.start(data),202)
            elif path=='/api/dataset-import/install':
                self.json(self.server.datasets.start({},install=True),202)
            elif path=='/api/dataset-import/cancel':
                self.json(self.server.datasets.cancel())
            elif path in ('/api/dataset-lyrics/scan','/api/dataset-lyrics/search','/api/dataset-lyrics/save'):
                from . import dataset_lyrics
                action = path.rsplit('/',1)[-1]
                self.json(getattr(dataset_lyrics,action)(data))
            elif path in ('/api/dataset-structure/scan','/api/dataset-structure/propose','/api/dataset-structure/apply'):
                from . import dataset_structure
                action = path.rsplit('/',1)[-1]
                if action == 'propose':
                    if not self.server.llm_lock.acquire(blocking=False):
                        self.json({'error':'The writing assistant is already working. Wait for its response.'},409)
                        return
                    try:
                        self.json(dataset_structure.propose(data))
                    finally:
                        self.server.llm_lock.release()
                else:
                    self.json(getattr(dataset_structure,action)(data))
            elif path=='/api/models/check':
                from .gguf import validate
                settings = validate_settings(data)
                try:
                    validate(settings,'audio')
                    self.json({'ready':True,'error':''})
                except ValueError as exc:
                    self.json({'ready':False,'error':str(exc)})
            elif path=='/api/models/download':
                with self.server.models.lock:
                    if any(j['status'] in ('queued','running','cancelling') for j in self.server.jobs.list()) or any(b['status'] in ('queued','writing','rendering','cancelling') for b in self.server.surprises.list()):
                        raise ValueError('Finish active songs and batches before downloading model files.')
                    self.json(self.server.models.start(data.get('variant')),202)
            elif path=='/api/models/cancel':
                self.json(self.server.models.cancel())
            elif path=='/api/surprises':
                with self.server.models.lock:
                    if self.server.models.busy(): raise ValueError('Wait for the model download to finish or cancel it first.')
                    self.json(self.server.surprises.start(data),202)
            elif re.fullmatch(r'/api/surprises/[a-f0-9]{32}/cancel',path):
                self.json(self.server.surprises.cancel(path.split('/')[3]))
            elif path=='/api/generate':
                with self.server.models.lock:
                    if self.server.models.busy(): raise ValueError('Wait for the model download to finish or cancel it first.')
                    self.json(self.server.jobs.generate(data),202)
            elif path=='/api/transcribe':
                self.json(self.server.jobs.transcribe(data),202)
            elif path=='/api/score':
                self.json(score_check(data.get('abc',''),data.get('strip',False),data.get('keep_voice','both')))
            elif path=='/api/loras/inspect':
                from .loras import inspect_adapter
                self.json(inspect_adapter(data.get('path')))
            elif path=='/api/artist-trainer/scan':
                from .artist_trainer import scan
                self.json(scan(data))
            elif path=='/api/artist-trainer/setup':
                self.json(self.server.jobs.prepare_artist(data),202)
            elif path=='/api/artist-trainer/train':
                self.json(self.server.jobs.train_artist(data),202)
            elif path=='/api/trainer/scan':
                from .trainer import scan
                self.json(scan(data))
            elif path=='/api/trainer/check':
                from .trainer_setup import readiness
                self.json(readiness(data))
            elif path=='/api/trainer/download':
                if set(data)!={'confirmed'} or data['confirmed'] is not True:
                    raise ValueError('Confirm the training-model download first.')
                self.json(self.server.jobs.download_training_models(),201)
            elif path=='/api/artist-trainer/projects':
                from .artist_trainer import save_project
                self.json(save_project(data),201)
            elif path=='/api/trainer/projects':
                from .trainer import save_project
                self.json(save_project(data),201)
            elif path=='/api/trainer/train':
                from .trainer_setup import readiness
                from .trainer import training_spec
                spec = training_spec(data)
                checked = readiness({key:spec[key] for key in ('model','vae')} | {'project_id':data['project_id']})
                if not checked['ready']:
                    raise ValueError(' '.join(checked['issues']))
                self.json(self.server.jobs.train(data),201)
            elif path=='/api/settings/validate':
                self.json({'settings':validate_settings(data)})
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/cancel',path):
                self.json(self.server.jobs.cancel(path.split('/')[3]))
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/star',path):
                if set(data) != {'starred'}:
                    raise ValueError('Expected only the starred field.')
                self.json(self.server.jobs.set_starred(path.split('/')[3],data['starred']))
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/rename',path):
                if set(data) != {'title'}:
                    raise ValueError('Expected only the title field.')
                self.json(self.server.jobs.rename(path.split('/')[3],data['title']))
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/delete',path):
                self.json(self.server.jobs.delete(path.split('/')[3]))
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/retry',path):
                original=self.server.jobs.detail(path.split('/')[3])
                if original['kind']!='generation' or original['status'] not in ('failed','cancelled','interrupted'):
                    raise ValueError('Only failed, cancelled or interrupted music runs can be retried.')
                spec=original['input']
                spec['source_job']=original['id']
                with self.server.models.lock:
                    if self.server.models.busy(): raise ValueError('Wait for the model download to finish or cancel it first.')
                    self.json(self.server.jobs.generate(spec),202)
            elif re.fullmatch(r'/api/jobs/[a-f0-9]{32}/wav',path):
                directory = self.server.jobs.directory(path.split('/')[3])
                source = directory/'result/audio.flac'
                if not source.is_file():
                    raise ValueError('This run has no rendered audio.')
                import soundfile as sf
                target = directory/'audio.wav'
                with self.server.jobs.lock:
                    if not target.exists():
                        temporary = directory/'audio.tmp.wav'
                        with sf.SoundFile(source) as incoming, sf.SoundFile(temporary,'w',samplerate=incoming.samplerate,channels=incoming.channels,subtype='PCM_24') as outgoing:
                            for block in incoming.blocks(blocksize=65536,dtype='float32'):
                                outgoing.write(block)
                        temporary.replace(target)
                self.json({'url':f'/artifacts/{directory.name}/audio.wav'})
            elif path=='/api/llm/models':
                self.json(llm.models(data))
            elif path in ('/api/llm/test','/api/llm/assist'):
                if not self.server.llm_lock.acquire(blocking=False):
                    self.json({'error':'The writing assistant is already working. Wait for its response.'},409)
                    return
                try:
                    self.json(llm.complete(data,'You are a helpful assistant.','Reply with a short connection confirmation.') if path.endswith('/test') else llm.assist(data))
                finally:
                    self.server.llm_lock.release()
            else:
                self.json({'error':'Not found.'},404)
        except (BrokenPipeError,ConnectionResetError):
            pass
        except PermissionError as exc:
            self.json({'error':str(exc)},403)
        except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
            self.json({'error':str(exc)},400)

    def upload(self):
        size = int(self.headers.get('Content-Length','0'))
        name = urllib.parse.unquote(self.headers.get('X-Filename','source.wav'))
        suffix = Path(name).suffix.lower()
        if suffix not in ('.wav','.flac','.mp3','.ogg','.m4a','.aac','.aiff','.aif','.opus'):
            raise ValueError('Choose WAV, FLAC, MP3, OGG, M4A, AAC, AIFF or OPUS audio.')
        if not 0<size<=300*1024*1024:
            raise ValueError('Source audio must be nonempty and no larger than 300 MB.')
        upload_id = uuid.uuid4().hex+suffix
        target = self.server.jobs.uploads/upload_id
        try:
            with target.open('xb') as handle:
                remaining = size
                while remaining:
                    block = self.rfile.read(min(1024*1024,remaining))
                    if not block:
                        raise ValueError('Audio upload was interrupted.')
                    handle.write(block)
                    remaining -= len(block)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        self.json({'upload_id':upload_id,'name':Path(name).name,'bytes':size,'url':'/uploads/'+upload_id},201)

    def file(self,path,download=False):
        if not path.is_file():
            self.json({'error':'File not found.'},404)
            return
        size = path.stat().st_size
        start,end = 0,size-1
        status = 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)',requested)
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            a,b = match.groups()
            if a:
                start=int(a)
                end=min(int(b),end) if b else end
            else:
                start=max(0,size-int(b))
            if start>end or start>=size:
                self.send_response(416)
                self.send_header('Content-Range',f'bytes */{size}')
                self.end_headers()
                return
            status=206
        self.send_response(status)
        mime = {'.js':'text/javascript','.flac':'audio/flac','.wav':'audio/wav','.svg':'image/svg+xml'}.get(path.suffix,mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Type',mime)
        self.send_header('Content-Length',str(max(0,end-start+1)))
        self.send_header('Accept-Ranges','bytes')
        if status==206:
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        if download:
            self.send_header('Content-Disposition',f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open('rb') as handle:
            handle.seek(start)
            remaining=end-start+1
            while remaining>0:
                block=handle.read(min(65536,remaining))
                if not block:
                    break
                self.wfile.write(block)
                remaining-=len(block)


def main():
    parser=argparse.ArgumentParser(description='YuE2 Studio — local music creation UI')
    parser.add_argument('--port',type=int,default=7862)
    parser.add_argument('--no-browser',action='store_true')
    parser.add_argument('--stop',action='store_true',help='Stop the studio listening on the selected port')
    parser.add_argument('--runs-dir',type=Path,default=ROOT/'runs/studio',help='Studio run and upload folder')
    args=parser.parse_args()
    if args.stop:
        url=f'http://127.0.0.1:{args.port}'
        try:
            with urllib.request.urlopen(url+'/api/bootstrap',timeout=2) as response:
                bootstrap=json.load(response)
            request=urllib.request.Request(url+'/api/shutdown',data=b'{}',
                headers={'Content-Type':'application/json','X-Studio-Token':bootstrap['token']})
            with urllib.request.urlopen(request,timeout=15) as response:
                json.load(response)
            print('Studio is stopping. Its active worker will be cancelled.',flush=True)
        except Exception as exc:
            parser.error(f'Could not stop Studio on port {args.port}: {exc}')
        return
    try:
        server=StudioServer(('127.0.0.1',args.port),root=args.runs_dir)
    except OSError as exc:
        url=f'http://127.0.0.1:{args.port}'
        try:
            with urllib.request.urlopen(url+'/api/bootstrap',timeout=2) as response:
                existing=json.load(response)
            if Path(existing['paths']['runs']).resolve()!=args.runs_dir.resolve():
                raise ValueError('Different studio run directory')
        except Exception:
            parser.error(f'Port {args.port} is unavailable. Choose another --port. {exc}')
        print(f'YuE2 Studio is already running: {url}',flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        return
    url=f'http://127.0.0.1:{server.server_port}'
    print(f'YuE2 Studio: {url}\nRuns: {server.jobs.root}\nPress Ctrl+C to stop.',flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.surprises.close()
        server.jobs.close()
        server.server_close()


if __name__=='__main__':
    main()

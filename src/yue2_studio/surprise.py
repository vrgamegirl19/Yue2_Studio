"""Server-owned sequential songwriting/render batches with request-scoped credentials."""
from copy import deepcopy
import json
import re
import secrets
import threading
import uuid

from . import llm
from .jobs import now, write_json
from .settings import validate_settings

VOICES = {'any':'Choose a suitable lead vocal gender.', 'female':'Female lead vocal.',
          'male':'Male lead vocal.', 'duet':'Female and male vocal duet.',
          'instrumental':'Instrumental only, no vocals.'}
IDEAS = [
    'an unexpected reunion', 'a small act of courage', 'leaving a familiar place',
    'finding humor in a bad day', 'a secret finally shared', 'a friendship across distance',
    'starting again after a setback', 'a celebration in an ordinary room',
    'a beautiful mistake', 'a change of heart', 'a road trip with no destination',
    'learning to let someone go', 'a promise kept years later', 'a restless summer evening',
    'the first morning in a new city', 'an unlikely connection', 'a well-earned victory',
    'a memory triggered by an everyday object', 'forgiving your younger self',
    'the excitement before a first meeting', 'an invitation to dance', 'a quiet rebellion',
]
ACTIVE = {'queued','writing','rendering','cancelling'}


class SurpriseManager:
    def __init__(self, jobs, llm_lock):
        self.jobs, self.llm_lock = jobs, llm_lock
        self.root = jobs.root/'batches'
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.records, self.events, self.threads = {}, {}, {}
        for path in self.root.glob('*.json'):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
                if record['status'] in ACTIVE:
                    record.update(status='interrupted',finished=now(),error='Studio restarted. Start a new batch; credentials were not saved.')
                    write_json(path,record)
                self.records[record['id']] = record
            except (OSError,ValueError,KeyError):
                continue

    def _save(self, record):
        write_json(self.root/(record['id']+'.json'),record)

    def list(self):
        with self.lock:
            return deepcopy(sorted(self.records.values(),key=lambda r:r['created'],reverse=True))

    def start(self, payload):
        if set(payload)-{'count','voice','style','language','brief','instructions','cot','settings','connection','profanity','lock_style'}:
            raise ValueError('Unknown Surprise me options.')
        lock_style=payload.get('lock_style',False)
        if type(lock_style) is not bool:
            raise ValueError('Keep style unchanged must be true or false.')
        if lock_style and (not isinstance(payload.get('style'),str) or not payload['style'].strip()):
            raise ValueError('Enter a style direction before locking it.')
        count=payload.get('count',1)
        voice=payload.get('voice','any')
        cot=payload.get('cot','full')
        profanity=payload.get('profanity','prompt')
        if profanity not in ('prompt','required'):
            raise ValueError('Choose a supported profanity setting.')
        if profanity=='required' and voice=='instrumental':
            raise ValueError('Profanity requires sung lyrics. Choose a vocal gender instead of Instrumental.')
        if type(count) is not int or not 1<=count<=50:
            raise ValueError('Choose a batch size from 1 to 50 songs.')
        if voice not in VOICES or cot not in ('full','melody','off'):
            raise ValueError('Choose a supported voice and generation mode.')
        settings=validate_settings(payload.get('settings',{}))
        if settings['runtime']['backend']=='audio.cpp':
            from .gguf import validate
            validate(settings,'audio')
        connection=deepcopy(payload.get('connection',{}))
        llm.config(connection)  # Validate before accepting a batch or spending tokens.
        options={k:str(payload.get(k) or '').strip() for k in ('style','language','brief','instructions')}
        if lock_style:options['style']=payload['style']  # Preserve exact supplied text.
        if any(len(v)>12000 for v in options.values()):
            raise ValueError('Shorten your surprise directions to 12,000 characters per field.')
        with self.lock:
            if self.jobs.stopping:
                raise ValueError('Studio is shutting down.')
            if any(r['status'] in ACTIVE for r in self.records.values()):
                raise ValueError('A surprise batch is already active. Finish or stop it before starting another.')
            record=dict(id=uuid.uuid4().hex,created=now(),status='queued',count=count,voice=voice,profanity=profanity,
                        cot=cot,lock_style=lock_style,options=options,settings=settings,provider=connection['provider'],
                        model=connection['model'],songs=[],current=0)
            self.records[record['id']]=record
            event=threading.Event()
            self.events[record['id']]=event
            self._save(record)
            thread=threading.Thread(target=self._run,args=(record['id'],connection,event),daemon=True)
            self.threads[record['id']]=thread
            thread.start()
            return deepcopy(record)

    def cancel(self, batch_id):
        with self.lock:
            record=self.records.get(batch_id)
            if record is None:
                raise ValueError('Unknown surprise batch.')
            if record['status'] in ACTIVE:
                # Cancel the owned render only. Other manually submitted jobs are untouched.
                if record.get('active_job'):
                    self.jobs.cancel(record['active_job'])
                self.events[batch_id].set()
                record['status']='cancelling'
                self._save(record)
            return deepcopy(record)

    def _update(self, batch_id, **changes):
        with self.lock:
            record=self.records[batch_id]
            record.update(changes)
            self._save(record)

    def _run(self, batch_id, connection, event):
        record=self.records[batch_id]
        try:
            previous=[]
            ideas=list(IDEAS)
            secrets.SystemRandom().shuffle(ideas)
            for index in range(record['count']):
                if event.is_set():break
                # Wait for our preceding render and for other active GPU work before writing.
                while any(j['status'] in ('queued','running','cancelling') for j in self.jobs.list()):
                    if event.wait(.5):break
                if event.is_set():break
                self._update(batch_id,status='writing',current=index+1,active_job=None)
                options=record['options']
                brief=(f'Create original song {index+1} of {record["count"]}. Fully invent its title, lyrics and musical style. '
                       'Return a complete, singable song with a developed second verse, repeated chorus and intentional ending. '
                       f'Vocal requirement: {VOICES[record["voice"]]} '
                       f'Language: {options["language"] or "Choose freely"}. '
                       f'Style constraint: {options["style"] or "Choose a fresh, coherent genre and arrangement"}. '
                       f'User direction: {options["brief"] or "Surprise me"}. '
                       f'Optional creative starting point: {ideas[index % len(ideas)]}. '
                       f'Variation seed: {secrets.token_hex(8)}. '
                       'Make this song distinct from the earlier songs below: new title, story, imagery and hook. '
                       'Honor the chosen voice and style; do not add contradictory vocal directions. '
                       f'Earlier songs: {json.dumps(previous,ensure_ascii=False)}')
                if record.get('lock_style'):
                    brief += ' STYLE IS LOCKED: invent only the title and lyrics to fit the supplied style. Do not redesign the genre, arrangement, or vocal character. The supplied style takes precedence over conflicting vocal directions; any style you return will be ignored.'
                if record.get('profanity')=='required':
                    brief += ' '+PROFANITY_DIRECTION
                acquired=False
                try:
                    while not event.is_set():
                        acquired=self.llm_lock.acquire(timeout=.5)
                        if acquired:break
                    if event.is_set():break
                    result=llm.assist({'connection':connection,'action':'song','brief':brief,
                                       'instructions':options['instructions']})
                finally:
                    if acquired:self.llm_lock.release()
                if event.is_set():break
                draft=result.get('draft')
                if result.get('truncated') or not draft:
                    raise ValueError('The LLM returned an incomplete draft. Increase its output-token limit or change model, then start a new batch.')
                if not draft['title'].strip() or (not record.get('lock_style') and not draft['style'].strip()):
                    raise ValueError('The LLM returned an empty title or style; this batch was stopped.')
                if record['voice']!='instrumental' and (not draft['lyrics'].strip() or '[' not in draft['lyrics']):
                    raise ValueError('The LLM returned empty or unsectioned lyrics; this batch was stopped.')
                if record.get('profanity')=='required' and profanity_count(draft['lyrics'])<3:
                    raise ValueError('The writing model did not include the required profanity (at least 3 uncensored strong English swear words in sung lines). No audio was queued. Change the writing instructions or model and try again.')
                if any(draft['title'].strip().casefold()==p['title'].casefold() or
                       (record['voice']!='instrumental' and draft['lyrics'].strip()==p['lyrics']) for p in previous):
                    raise ValueError('The LLM repeated an earlier song. This batch was stopped to avoid rendering duplicates.')
                style=options['style'] if record.get('lock_style') else draft['style']
                if not record.get('lock_style') and record['voice']!='any':
                    if record['voice'] == 'instrumental':
                        # Do not use negative words like 'no vocals' (attention leakage)
                        style = 'pure instrumental, ' + style
                    else:
                        style = VOICES[record['voice']] + ' ' + style
                lyrics='' if record['voice']=='instrumental' else draft['lyrics']
                cot_mode = 'off' if record['voice']=='instrumental' else record['cot']
                spec={'title':draft['title'],'mode':'create','stage':'audio',
                      'source_job':f'surprise:{batch_id}:{index+1}', 'settings':record['settings'],
                      'request':{'style':style,'lyrics':lyrics,'cot':cot_mode,'seed':secrets.randbits(63)}}
                # Lock prevents a stop request from slipping between submission and ownership.
                with self.lock:
                    if event.is_set():break
                    job=self.jobs.generate(spec)
                    write_json(self.jobs.directory(job['id'])/'songwriting.json',
                               {'draft':draft,'provider':record['provider'],'model':record['model'],
                                'batch_id':batch_id,'index':index+1,'voice':record['voice']})
                    record['songs'].append({'id':job['id'],'title':draft['title'],'status':'queued'})
                    record.update(status='rendering',active_job=job['id'])
                    self._save(record)
                previous.append({'title':draft['title'].strip(),'lyrics':draft['lyrics'].strip()})
                while True:
                    status=next(j['status'] for j in self.jobs.list() if j['id']==job['id'])
                    if status not in ('queued','running','cancelling'):break
                    event.wait(.5)
                    if event.is_set():
                        self.jobs.cancel(job['id'])
                        break
                with self.lock:
                    record['songs'][-1]['status']=status
                    self._save(record)
                if event.is_set():break
                if status not in ('complete','needs_review'):
                    raise ValueError('A song render '+status+'. The remaining batch was stopped; inspect that run in the library.')
            self._update(batch_id,status='cancelled' if event.is_set() else 'complete',finished=now(),active_job=None)
        except Exception as exc:
            error=str(exc)
            secret=connection.get('api_key')
            if secret:error=error.replace(secret,'[redacted]')
            self._update(batch_id,status='cancelled' if event.is_set() else 'failed',error=error,finished=now(),active_job=None)
        finally:
            connection.clear()

    def close(self):
        for record in self.list():
            if record['status'] in ACTIVE:self.cancel(record['id'])


PROFANITY_DIRECTION = (
    'PROFANITY REQUIREMENT: Include at least 3 occurrences of uncensored strong English swear words '
    'in the actual sung lines, such as fuck, fucking, shit, bullshit, bitch or motherfucker. '
    'Use them naturally for emphasis and emotional impact; dark themes alone do not satisfy this. '
    'No asterisks, bleeps, euphemisms or clean substitutions. Words in tags, style or notes do not count. '
    'Check this requirement before returning JSON. Keep the lyrics original and singable.')


def profanity_count(lyrics):
    sung = re.sub(r'\[[^\]]*\]', '', lyrics)
    return len(re.findall(r"\b(?:fuck(?:s|ed|ing|er|ers)?|motherfuck(?:er|ers|ing)?|shit(?:s|ty|ting)?|bullshit|bitch(?:es|ing)?|asshole(?:s)?)\b", sung, re.I))

"""CPU-only queue validation for the separate artist AR trainer."""
import math
from . import artist_trainer
from .settings import ROOT

REG_REVISION='5d00559c3daa5cfb7a61fbe32158c8c08f9b5f35'
REG_SHA256='bdd9b9780de46bb0752c3e3bc101869759493443c6e35b1b2d4a2c5033eadc4e'


def validate_project(project):
    inspected=artist_trainer.scan(project)
    current={r['name']:r for r in inspected['tracks']}
    selected=[r for r in project['tracks'] if r['enabled']]
    if len(selected)<2:
        raise ValueError('Select at least two songs: one must be held out for validation.')
    for row in selected:
        fresh=current.get(row['name'])
        if not fresh or fresh['error'] or any(row[k]!=fresh[k] for k in ('bytes','mtime_ns','lyrics_sha256')):
            raise ValueError('Selected audio or lyrics changed. Rescan and save a new artist setup.')
        if not 30<=row['seconds']<=360:
            raise ValueError('This experimental artist trainer supports 30–360 second recordings; exclude '+row['name'])
    if not project.get('shared_style','').strip() or not project.get('trigger','').strip():
        raise ValueError('The saved artist setup needs a shared style and trigger.')
    return selected


def training_spec(payload):
    allowed={'project_id','steps','rank','learning_rate','checkpoint_every','alignment_weight','alignment_method','gpu_confirmed'}
    if set(payload)-allowed:raise ValueError('Unknown artist training controls.')
    if payload.get('gpu_confirmed') is not True:
        raise ValueError('Confirm GPU preparation and training before queueing.')
    project=artist_trainer.load_project(str(payload.get('project_id','')))
    selected=validate_project(project)
    controls={}
    for key,default,lo,hi in [('steps',500,1,1600),('rank',64,8,64),('checkpoint_every',250,1,400)]:
        value=payload.get(key,default)
        if type(value) is not int or not lo<=value<=hi:raise ValueError(f'{key} must be an integer from {lo} to {hi}.')
        controls[key]=value
    for key,default,lo,hi in [('learning_rate',1e-4,1e-6,3e-4),('alignment_weight',.08,0,.2)]:
        value=payload.get(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+key)
        controls[key]=value
    method=payload.get('alignment_method','mms')
    if type(method) is not str or method not in ('mms','whisper'):
        raise ValueError('Choose MMS or Whisper-assisted lyric timing.')
    controls['alignment_method']=method
    from .artist_setup import check_setup
    setup=check_setup()
    if not setup['files_and_imports_ready']:raise ValueError('Artist setup incomplete: '+'; '.join(setup['issues']))
    runtime=setup['runtime']['python']
    if method=='whisper' and controls['alignment_weight']>0:
        from .artist_setup import check_whisper_runtime
        check_whisper_runtime(runtime)
    return dict(title=project['name']+' · Artist LoRA',stage='artist_train',mode='artist_trainer',
                project=project,controls=controls,python=str(runtime),paths=setup['paths'],holdout=selected[-1]['name'],
                regularizer_revision=REG_REVISION,regularizer_sha256=REG_SHA256,gpu_confirmed=True)

"""Provider adapters; credentials are request-scoped and never persisted."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request

PROMPT = (Path(__file__).parent / 'prompts/songwriter.md').read_text(encoding='utf-8')
PROVIDERS = [
    dict(id='openai', label='OpenAI', url='https://api.openai.com/v1', protocol='responses', key=True),
    dict(id='anthropic', label='Anthropic', url='https://api.anthropic.com/v1', protocol='anthropic', key=True),
    dict(id='google', label='Google Gemini', url='https://generativelanguage.googleapis.com/v1beta', protocol='google', key=True),
    dict(id='grok', label='Grok · xAI', url='https://api.x.ai/v1', protocol='chat', key=True),
    dict(id='deepseek', label='DeepSeek', url='https://api.deepseek.com/v1', protocol='chat', key=True),
    dict(id='openrouter', label='OpenRouter', url='https://openrouter.ai/api/v1', protocol='chat', key=True),
    dict(id='apifreellm', label='APIFreeLLM', url='https://apifreellm.com/api/v1', protocol='apifreellm', key=True),
    dict(id='groq', label='Groq', url='https://api.groq.com/openai/v1', protocol='chat', key=True),
    dict(id='lm_studio', label='LM Studio', url='http://127.0.0.1:1234/v1', protocol='lm_studio', key=False),
    dict(id='ollama', label='Ollama', url='http://127.0.0.1:11434', protocol='ollama', key=False),
    dict(id='own_server', label='Custom server', url='http://127.0.0.1:8000/v1', protocol='chat', key=False),
]
LOCAL = {'lm_studio','ollama','own_server'}


def catalogue():
    path = Path(__file__).parent / 'provider_models.json'
    fallback = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    return [{**p, 'models': fallback.get(p['id'], [])} for p in PROVIDERS]


def api_root(value, add_v1=True):
    parsed = urllib.parse.urlsplit(str(value).strip())
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Server URL must be HTTP(S), without credentials, a query or fragment.')
    root = str(value).strip().rstrip('/')
    for ending in ('/chat/completions','/models','/responses'):
        if root.endswith(ending):
            root = root[:-len(ending)]
    if add_v1 and not root.endswith('/v1'):
        root += '/v1'
    return root


def config(value, need_model=True):
    if not isinstance(value,dict):
        raise ValueError('LLM connection settings are missing.')
    provider = next((p for p in PROVIDERS if p['id'] == value.get('provider')), None)
    if provider is None:
        raise ValueError('Choose an LLM provider.')
    key = str(value.get('api_key') or '').strip()
    if provider['key'] and not key:
        raise ValueError(f"Enter your {provider['label']} API key in LLM Runner.")
    if '\n' in key or '\r' in key:
        raise ValueError('Invalid API key.')
    model = str(value.get('model') or '').strip()
    if need_model and (not model or len(model)>256):
        raise ValueError('Select a model or enter its exact ID in LLM Runner.')
    root = provider['url']
    if provider['id'] in LOCAL:
        root = api_root(value.get('base_url') or root, provider['id'] != 'ollama')
    timeout = value.get('timeout',360)
    limit = value.get('max_tokens',4096)
    context = value.get('context_length',32768)
    temperature = value.get('temperature')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 15<=timeout<=600:
        raise ValueError('LLM timeout must be 15–600 seconds.')
    if type(limit) is not int or not 64<=limit<=131072:
        raise ValueError('Maximum output tokens must be an integer from 64 to 131072.')
    if type(context) is not int or not 512<=context<=1048576:
        raise ValueError('Local context must be an integer from 512 to 1048576.')
    if temperature is not None and (type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=2):
        raise ValueError('Writing temperature must be blank or between 0 and 2.')
    headers = {'Content-Type':'application/json','Accept':'application/json','User-Agent': 'LLM-Runner/1.0'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    if provider['protocol'] == 'anthropic':
        headers = {'Content-Type':'application/json','x-api-key':key,'anthropic-version':'2023-06-01'}
    if provider['protocol'] == 'google':
        headers = {'Content-Type':'application/json','x-goog-api-key':key}
    return {**provider,'url':root,'headers':headers,'model':model,'timeout':timeout,
            'max_tokens':limit,'context_length':context,'temperature':temperature,'secret':key}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('The LLM endpoint redirected. Enter the final API URL; credentials were not forwarded.')


def request_json(url, cfg, payload=None):
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode('utf-8'),
                                     headers=cfg['headers'],method='GET' if payload is None else 'POST')
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=cfg['timeout']) as response:
            body = response.read(12*1024*1024+1)
            if len(body)>12*1024*1024:
                raise ValueError('LLM response is too large.')
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        raw = exc.read(12000).decode('utf-8','replace')
        if cfg['secret']:
            raw = raw.replace(cfg['secret'],'[redacted]')
        try:
            error = json.loads(raw).get('error',raw)
            raw = error.get('message',str(error)) if isinstance(error,dict) else str(error)
        except (ValueError,AttributeError):
            raw = f'HTTP {exc.code}. Check the endpoint, model access, API key and provider limits.'
        raise ValueError(f"{cfg['label']} HTTP {exc.code}: {raw[:900]}") from None
    except (urllib.error.URLError,TimeoutError) as exc:
        reason = str(getattr(exc,'reason',exc))
        if cfg['secret']:
            reason = reason.replace(cfg['secret'],'[redacted]')
        raise ValueError(f"Could not reach {cfg['label']}: {reason}") from None


def models(value):
    cfg = config(value,need_model=False)
    protocol,root = cfg['protocol'],cfg['url']
    if protocol == 'apifreellm':
        return {'models':['apifreellm'],'source':'Provider uses a fixed model; connection has not been tested.'}
    if protocol == 'ollama':
        data = request_json(root+'/api/tags',cfg)
        ids = [x['name'] for x in data.get('models',[])]
    else:
        ids = []
        query = ''
        for _ in range(30):
            data = request_json(root+'/models'+query,cfg)
            if protocol == 'google':
                ids.extend(x['name'].removeprefix('models/') for x in data.get('models',[]) if 'generateContent' in x.get('supportedGenerationMethods',[]))
                page = data.get('nextPageToken')
                query = '?pageToken='+urllib.parse.quote(page,safe='') if page else ''
            else:
                ids.extend(x['id'] for x in data.get('data',[]) if isinstance(x.get('id'),str))
                page = data.get('last_id') if data.get('has_more') else None
                query = '?after_id='+urllib.parse.quote(page,safe='') if page else ''
            if not query:
                break
    if cfg['id']=='openai':
        ids = [m for m in ids if not any(x in m.lower() for x in ('embedding','image','audio','tts','whisper','transcribe','realtime','moderation','sora'))]
    return {'models':sorted(set(ids),key=str.lower),'source':'Live provider model list. Select a text-generation model; availability does not guarantee endpoint compatibility.'}


def content_text(content):
    if isinstance(content,str):
        return content
    if isinstance(content,list):
        return ''.join(x.get('text','') for x in content if isinstance(x,dict) and x.get('type') in ('text','output_text'))
    return ''


def complete(value, system, user):
    cfg = config(value)
    root,protocol = cfg['url'],cfg['protocol']
    temp = {} if cfg['temperature'] is None else {'temperature':cfg['temperature']}
    messages = [{'role':'system','content':system},{'role':'user','content':user}]
    limit,model = cfg['max_tokens'],cfg['model']
    if protocol == 'responses':
        data = request_json(root+'/responses',cfg,{'model':model,'instructions':system,'input':user,'max_output_tokens':limit,'store':False,**temp})
        text = ''.join(content_text(x.get('content')) for x in data.get('output',[]) if x.get('type')=='message')
        stop = data.get('status','')
        truncated = stop=='incomplete'
    elif protocol == 'anthropic':
        data = request_json(root+'/messages',cfg,{'model':model,'system':system,'messages':messages[1:],'max_tokens':limit,**temp})
        text = content_text(data.get('content'))
        stop = data.get('stop_reason','')
        truncated = stop=='max_tokens'
    elif protocol == 'google':
        data = request_json(root+'/models/'+urllib.parse.quote(model.removeprefix('models/'),safe='')+':generateContent',cfg,
            {'systemInstruction':{'parts':[{'text':system}]},'contents':[{'role':'user','parts':[{'text':user}]}],
             'generationConfig':{'maxOutputTokens':limit,**temp}})
        candidate = next(iter(data.get('candidates',[])),{})
        text = ''.join(x.get('text','') for x in candidate.get('content',{}).get('parts',[]) if not x.get('thought'))
        stop = candidate.get('finishReason','')
        truncated = stop=='MAX_TOKENS'
    elif protocol == 'ollama':
        data = request_json(root+'/api/chat',cfg,{'model':model,'messages':messages,'stream':False,
            'options':{'num_predict':limit,'num_ctx':cfg['context_length'],**temp},'keep_alive':0})
        text = data.get('message',{}).get('content','')
        stop = data.get('done_reason','')
        truncated = stop=='length'
    elif protocol == 'lm_studio':
        data = request_json(root.removesuffix('/v1')+'/api/v1/chat',cfg,{'model':model,'system_prompt':system,'input':user,
            'context_length':cfg['context_length'],'max_output_tokens':limit,**temp})
        text = ''.join(x.get('content','') for x in data.get('output',[]) if x.get('type')=='message')
        stop = data.get('stop_reason','')
        truncated = stop in ('length','max_tokens')
    elif protocol == 'apifreellm':
        if temp or limit != 4096:
            raise ValueError('APIFreeLLM does not expose output-token or temperature controls in its runner. Use automatic temperature and the default 4096 setting (not sent).')
        data = request_json(root+'/chat',cfg,{'model':model,'message':system+'\n\n'+user})
        text = next((data[k] for k in ('response','message','text','output','content') if isinstance(data.get(k),str)), '')
        stop,truncated = '',False
    else:
        data = request_json(root+'/chat/completions',cfg,{'model':model,'messages':messages,'max_tokens':limit,**temp})
        choice = next(iter(data.get('choices',[])),{})
        text = content_text(choice.get('message',{}).get('content'))
        stop = choice.get('finish_reason','')
        truncated = stop=='length'
    if not text.strip():
        raise ValueError(f"{cfg['label']} returned no text (status: {stop or 'empty'}). Check the chosen model and output-token limit.")
    return dict(text=text.strip(),provider=cfg['id'],model=model,truncated=truncated,finish_reason=stop)


def assist(payload):
    action = payload.get('action','song')
    if action not in ('song','lyrics','style','review','adapt'):
        raise ValueError('Unknown songwriting action.')
    brief = str(payload.get('brief') or '').strip()
    if not brief:
        raise ValueError('Describe the song or the changes you want.')
    custom = str(payload.get('instructions') or '')
    context = {key:str(payload.get(key) or '') for key in ('title','style','lyrics','abc')}
    if sum(map(len,context.values()))+len(brief)+len(custom)>180000:
        raise ValueError('Songwriting context is too large. Shorten your brief or score.')
    user = json.dumps({'task':action,'brief':brief,'current_song':context},ensure_ascii=False)
    result = complete(payload.get('connection'),PROMPT+ ('\n\nUSER WRITING PREFERENCES\n'+custom if custom else ''),user)
    raw = re.sub(r'^```(?:json)?\s*|\s*```$','',result['text'],flags=re.I).strip()
    try:
        draft = json.loads(raw)
        if not isinstance(draft,dict) or any(not isinstance(draft.get(k),str) for k in ('title','style','lyrics','notes')):
            raise ValueError('Wrong response fields')
        result['draft'] = {k:draft[k] for k in ('title','style','lyrics','notes')}
    except (ValueError,TypeError):
        result['draft'] = None
        result['warning'] = 'The model did not return a complete structured draft. The raw response is preserved below. Try again or increase the output-token limit.'
    return result

/* CPU-only check: Artist dropdown is present and reaches the training payload. */
const vm=require('node:vm'),fs=require('node:fs'),assert=require('node:assert/strict');
const elements=new Map(),calls=[];
function el(id){
  if(!elements.has(id))elements.set(id,{value:'',checked:false,textContent:'',options:[],
    replaceChildren(...items){this.options=items;},add(item){this.options.push(item);}});
  return elements.get(id);
}
const context=vm.createContext({$:el,Option:function(text,value){this.text=text;this.value=value;},
  toast:()=>{},state:{},openRun:async()=>{},poll:async()=>{},busy:async(_,fn)=>fn(),
  confirmReplace:(_,__,fn)=>fn(),document:{createElement:()=>({append(){}}),createTextNode:text=>text},
  api:async(url,payload)=>{
    calls.push({url,payload});
    if(url==='/api/artist-trainer/projects')return {projects:[]};
    if(url==='/api/artist-trainer/train')return {id:'queued'};
    throw new Error(url);
  }});
vm.runInContext(fs.readFileSync('src/yue2_studio/static/artist.js','utf8'),context);
context.bindArtistSetup=()=>{};
(async()=>{
  vm.runInContext('bindArtistTrainer()',context);
  el('artistProjects').value='saved';el('artistGpuConfirmed').checked=true;
  for(const [id,value] of [['artistSteps','20'],['artistRank','64'],['artistLearningRate','0.0001'],
    ['artistCheckpoint','20'],['artistAlignment','0.08'],['artistAlignmentMethod','whisper']])el(id).value=value;
  el('artistTrain').onclick();
  await new Promise(resolve=>setImmediate(resolve));
  const queued=calls.find(call=>call.url==='/api/artist-trainer/train');
  assert.equal(queued.payload.alignment_method,'whisper');
  assert.equal(queued.payload.project_id,'saved');
  assert.equal(el('artistGpuConfirmed').checked,false);
  const html=fs.readFileSync('src/yue2_studio/static/index.html','utf8');
  assert.match(html,/id="artistAlignmentMethod"/);
  assert.match(html,/<option value="mms">/);
  assert.match(html,/<option value="whisper">/);
  console.log('Artist Whisper dropdown and payload passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});

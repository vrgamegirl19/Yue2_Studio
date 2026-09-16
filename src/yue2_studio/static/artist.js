'use strict';
let artistData=null,artistDefaultFolder='';
function artistOptions(){return {folder:$('artistFolder').value.trim().replace(/^"(.*)"$/,'$1'),lyrics_suffix:$('artistSuffix').value};}
function artistSelectionChanged(){
  $('artistReviewed').checked=false;$('artistSaveStatus').textContent='Review this selection before saving.';
  const selected=artistData.tracks.filter(row=>row.enabled);
  $('artistSummary').textContent=selected.length+' of '+artistData.tracks.length+' pairs selected · '+(selected.reduce((sum,row)=>sum+row.seconds,0)/60).toFixed(1)+' minutes';
}
function renderArtist(){
  $('artistReview').hidden=!artistData;if(!artistData)return;
  const container=$('artistTracks');container.replaceChildren();
  for(const row of artistData.tracks){
    const card=document.createElement('article');card.className='trainer-track';
    const label=document.createElement('label');label.className='check-label';
    const check=document.createElement('input');check.type='checkbox';check.checked=row.enabled;check.disabled=Boolean(row.error);
    check.onchange=()=>{row.enabled=check.checked;artistSelectionChanged();};label.append(check,document.createTextNode(row.name));card.append(label);
    const meta=document.createElement('p');meta.className='hint';meta.textContent=row.error||row.seconds.toFixed(1)+' s · '+row.lyrics_name;card.append(meta);
    if(row.warnings.length){const warning=document.createElement('p');warning.className='hint';warning.textContent=row.warnings.join(' ');card.append(warning);}
    if(row.lyrics){const details=document.createElement('details'),summary=document.createElement('summary'),preview=document.createElement('pre');summary.textContent='Review complete lyrics';preview.textContent=row.lyrics;preview.style.whiteSpace='pre-wrap';details.append(summary,preview);card.append(details);}
    container.append(card);
  }
  artistSelectionChanged();
}
async function refreshArtist(){
  
  try{
    const data=await api('/api/artist-trainer/projects');
    
    const select=$('artistProjects'),previous=select.value;select.replaceChildren(new Option(data.projects.length?'Choose an artist setup':'No saved artist setups',''));
    for(const project of data.projects)select.add(new Option(project.name+' · '+new Date(project.created).toLocaleString(),project.id));
    if(data.projects.some(p=>p.id===previous))select.value=previous;
  }catch(error){$('artistStatus').textContent=error.status===404?'Artist setup needs a Studio restart after your renders finish. Do not restart during a render.':error.message;}
}
function bindArtistTrainer(){
  bindArtistSetup();
  $('artistTrain').onclick=()=>{
    const project_id=$('artistProjects').value;
    if(!project_id){toast('Choose a saved artist setup first.',true);return;}
    if(!$('artistGpuConfirmed').checked){toast('Confirm GPU preparation and training first.',true);return;}
    const payload={project_id,gpu_confirmed:true,steps:Number($('artistSteps').value),rank:Number($('artistRank').value),learning_rate:Number($('artistLearningRate').value),checkpoint_every:Number($('artistCheckpoint').value),alignment_weight:Number($('artistAlignment').value),alignment_method:$('artistAlignmentMethod').value};
    confirmReplace('Queue Artist LoRA training?','Uses the selected saved setup, not unsaved edits. '+(payload.alignment_weight===0?'Lyric timing is skipped.':payload.alignment_method==='whisper'?'Whisper-assisted lyric timing will save a per-song review JSON.':'MMS lyric alignment will run.')+' Whole-song encoding runs before '+payload.steps+' training steps in the shared GPU queue. Playback requires No score, Torch, quantization None, and AR offloading disabled.',()=>busy('artistTrain',async()=>{
      const job=await api('/api/artist-trainer/train',payload);$('artistGpuConfirmed').checked=false;
      $('artistTrainStatus').textContent='Queued. Open the Library run for preparation progress, training steps, logs, and checkpoints.';
      state.activeId=job.id;await openRun(job.id);await poll();
    }));
  };

  const scan=()=>busy('artistScan',async()=>{
    artistData=await api('/api/artist-trainer/scan',artistOptions());$('artistFolder').value=artistData.folder;
    if(!$('artistName').value)$('artistName').value=artistData.metadata.singer_name||'Artist experiment';
    if(!$('artistStyle').value)$('artistStyle').value=artistData.metadata.style_prompt||'';
    $('artistStatus').textContent=artistData.note;renderArtist();
  });
  $('artistScan').onclick=()=>artistData?confirmReplace('Rescan artist dataset?','Song selections and the review confirmation will reset. Your style stays in the form.',scan):scan();
  $('artistSelectAll').onclick=()=>{for(const row of artistData.tracks)row.enabled=!row.error;renderArtist();};
  $('artistSelectNone').onclick=()=>{for(const row of artistData.tracks)row.enabled=false;renderArtist();};
  $('artistSave').onclick=()=>busy('artistSave',async()=>{
    if(!artistData)throw new Error('Scan the artist dataset first.');
    const options=artistOptions();if(options.folder!==artistData.folder||options.lyrics_suffix!==artistData.lyrics_suffix)throw new Error('Folder or lyric naming changed. Scan again.');
    const saved=await api('/api/artist-trainer/projects',{...options,name:$('artistName').value,trigger:$('artistTrigger').value,shared_style:$('artistStyle').value,lyrics_reviewed:$('artistReviewed').checked,
      tracks:artistData.tracks.map(({name,bytes,mtime_ns,lyrics_name,lyrics_sha256,lyrics_bytes,lyrics_mtime_ns,enabled})=>({name,bytes,mtime_ns,lyrics_name,lyrics_sha256,lyrics_bytes,lyrics_mtime_ns,enabled}))});
    $('artistSaveStatus').textContent='Saved '+saved.name+'. Setup only; no encoding or training started.';await refreshArtist();$('artistProjects').value=saved.id;
  });
  $('artistOpen').onclick=()=>{
    const id=$('artistProjects').value;if(!id)return;
    const open=()=>busy('artistOpen',async()=>{
      artistData=await api('/api/artist-trainer/projects/'+id);
      for(const [field,key] of [['artistName','name'],['artistTrigger','trigger'],['artistStyle','shared_style'],['artistFolder','folder'],['artistSuffix','lyrics_suffix']])$(field).value=artistData[key];
      renderArtist();$('artistStatus').textContent='Saved snapshot loaded. Sources are rechecked on Save; scan again to refresh previews.';
    });
    if(artistData)confirmReplace('Open artist setup?','This replaces the current artist form. Save it first to keep edits.',open);else open();
  };
  $('artistRefresh').onclick=refreshArtist;refreshArtist();
}

const artistPathKeys=['model','vae','mert','encoder','regularizer'];
function artistSetupPaths(){return Object.fromEntries(artistPathKeys.map(key=>[key,$('artistPath_'+key).value.trim()]).filter(([,value])=>value));}
function bindArtistSetup(){
  const container=$('artistModelPaths');
  for(const key of artistPathKeys){
    const label=document.createElement('label');label.htmlFor='artistPath_'+key;label.textContent='Local '+key+' folder';
    const input=document.createElement('input');input.id='artistPath_'+key;input.spellcheck=false;
    container.append(label,input);
  }
  api('/api/artist-trainer/paths').then(data=>{
    for(const key of artistPathKeys)if(!$('artistPath_'+key).value)$('artistPath_'+key).value=data.paths[key]||'';
  }).catch(error=>{$('artistRuntimeStatus').textContent=error.message;});
  const queue=async(action,confirmed=false)=>{
    const job=await api('/api/artist-trainer/setup',{action,confirmed,terms_accepted:$('artistTerms').checked,paths:artistSetupPaths()});
    $('artistRuntimeStatus').textContent='Queued '+action+'. See its Library run for results. No training started.';
    await openRun(job.id);
  };
  $('artistCheck').onclick=()=>busy('artistCheck',()=>queue('check'));
  $('artistDownload').onclick=()=>{
    if(!$('artistTerms').checked){toast('Review and accept the model terms first.',true);return;}
    confirmReplace('Prepare Artist model paths?','Reuses verified local files and downloads missing models into a separate cache. Downloads may total many GB. Existing invalid folders are not overwritten. This does not start training.',()=>busy('artistDownload',()=>queue('download-models',true)));
  };
  $('artistInstall').onclick=()=>confirmReplace('Install separate Artist dependencies?','Downloads a separate Torch runtime and dependencies (several GB). Leaves Studio’s current Python environment unchanged. This does not start training.',()=>busy('artistInstall',()=>queue('install-runtime',true)));
}

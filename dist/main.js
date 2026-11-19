import {Fluid} from './fluid.js';
import {Renderer, cameraRay} from './render.js';
import {FrameTimings} from './timing.js';
import {Fallback} from './fallback.js';
import {Fluid as OriginalFluid} from './original/fluid.js';
import {Renderer as OriginalRenderer} from './original/render.js';
import {Fallback as OriginalFallback} from './original/fallback.js';

import {Fluid as EnhancedFluid} from './enhanced/fluid.js';
import {Renderer as EnhancedRenderer} from './enhanced/render.js';
import {Fallback as EnhancedFallback} from './enhanced/fallback.js';

import {CpuVolume} from './enhanced/cpu-volume.js';
import {presets} from './presets.js';

const models={
  enhanced:{Fluid:EnhancedFluid, Renderer:EnhancedRenderer, Fallback:EnhancedFallback},
  original:{Fluid:OriginalFluid, Renderer:OriginalRenderer, Fallback:OriginalFallback},
  revised:{Fluid, Renderer, Fallback}
};
let selectedModel=models.enhanced, activeModel=selectedModel;

const el = id => document.getElementById(id);
const stage = el('viewport');
let canvas = el('view');
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const controls = ['fuel','wind','turbulence','oxygen','exposure'];
const params = {preset:'wood', ...presets.wood, brush:null, coldStart:false, expansion:true};
const compact = matchMedia('(max-width: 760px)').matches;
const cameraDefault = {yaw:.38, pitch:.14, distance:3.9, targetHeight:.98};
const opts = {...cameraDefault, exposure:1, quality:compact?'low':'medium', mode:'beauty',stableSampling:true,sharedLighting:true};
const renderPasses=['lighting','surface','volume','resolve'];
let gridQuality=opts.quality;
const sceneSettings = {};
const sceneCameras = {};
function defaultSceneCamera(){
  const preset=presets[params.preset];
  const camera={...cameraDefault,...preset.camera};
  if(compact && preset.compactDistance)camera.distance=preset.compactDistance;
  return camera;
}
let timings, gl, engine, renderer, isGPU=false, dirty=true;
let cpuVolume=false;
const spatial=()=>isGPU || cpuVolume;
function currentBackend(){
  if(isGPU)return 'gpu';
  return cpuVolume?'cpu-volume':'cpu';
}
function createCpuEngine(){
  return cpuVolume?new CpuVolume(canvas,opts.quality):new selectedModel.Fallback(canvas);
}
const gpuFences=[];
const gpuStallLimit=4000;
let gpuBlockedSince=null;
let activePreview=null, solverInterrupted=false;
let stateRevision=0, renderSerial=0, pendingSteps=0, pendingCapture=false, pendingBrush=null;
let time=0, accumulator=0, running=true, speed=1, last=0, lastStats=0, frameCount=0, stepCount=0, elapsed=0;
const enhancedStep=1/60, enhancedDebtLimit=.25;
let simulatedSeconds=0, droppedClockTime=0;
let playbackMeasurement=null;
let brushMode=false, dragging=false, activeBrush=false, toastTimer, pendingReset=false;
let renderScale=1, slowWindows=0, fastWindows=0, skipTimingUntil=0;
const autoResolution=el('auto-resolution');
const legacyScales=[1,.92,.84], enhancedScales=[1,.92,.84,.70];
let resolutionTrial=null, blockedFrames=0;
function resetAdaptiveResolution() {
  const shortEvent=selectedModel===models.enhanced && presets[params.preset].oneShot;
  renderScale=shortEvent && autoResolution.checked && (opts.quality==='low' || opts.quality==='medium')?(compact?.84:.92):1;
  slowWindows=0;fastWindows=0;resolutionTrial=null;blockedFrames=0;
  skipTimingUntil=performance.now()+(shortEvent?0:2000);
}

const frameGaps=[];
let lastPresentation=0;
let sparkRemaining=0;
const sparkPositions = {wood:[.5,.06,.5], crosswind:[.5,.06,.5], pool:[.5,.07,.5], jet:[.5,.07,.5]};
function timestep() {
  if(activeModel===models.enhanced)return enhancedStep;
  return isGPU && (opts.quality==='high' || opts.quality==='ultra') ? 1/60 : 1/30;
}
const activePointers=new Map();
const recentFrames=[];
const select=el('quality');
const isWoodScene=()=>params.preset==='wood' || params.preset==='crosswind';

function toast(text) {
  const field=el('toast');
  field.textContent=text;
  clearTimeout(toastTimer);
  toastTimer=setTimeout(()=>field.textContent='',2400);
}
function syncPlayback() {
  const preview=activePreview?.open;
  el('live').textContent=solverInterrupted?'Interrupted':running?(preview?'Preview open':'Live'):'Paused';
  el('live-badge').classList.toggle('paused',!running || Boolean(preview));
  el('pause').querySelector('use').setAttribute('href',running?'#i-pause':'#i-play');
  el('pause').setAttribute('aria-label',running?'Pause simulation':'Play simulation');
  if(!running || preview)el('fps').textContent='—';
  updateHint();
}
function resetPlaybackClock() {
  finishMeasurement();
  cancelFrame();
  accumulator=0; last=0; lastStats=performance.now();
  frameCount=0;stepCount=0;elapsed=0;simulatedSeconds=0;lastPresentation=0;frameGaps.length=0;
  gpuBlockedSince=null;throttledDisplay=false;
}
function setPaused(pause) {
  running=!pause;
  resetPlaybackClock();
  syncPlayback();
  timer();
  if(isGPU && renderScale!==1)dirty=true;
  scheduleFrame();
}
function timer() {
  const centis=Math.floor(time*100+.001);
  el('sim-time').textContent=String(Math.floor(centis/6000)).padStart(2,'0')+':'+String(Math.floor(centis/100)%60).padStart(2,'0')+'.'+String(centis%100).padStart(2,'0');
}
function syncControls() {
  el('fuel-label').textContent=selectedModel===models.revised && isWoodScene()?'Fuel loading':'Fuel feed';
  for(const name of controls) {
    const input=el(name);
    const v=name==='exposure'?opts.exposure:params[name];
    input.value=String(v);
    const label=name==='exposure'?v.toFixed(2)+'×':name==='wind'?(v>0?'+':'')+v.toFixed(2):Math.round(v*100)+'%';
    el(name+'-value').textContent=label;
    input.style.setProperty('--fill',100*(v-Number(input.min))/(Number(input.max)-Number(input.min))+'%');
  }
}
function queueRestart(coldStart=params.coldStart) {
  finishMeasurement();
  time=0; accumulator=0; last=0; droppedClockTime=0; params.brush=null;
  timings?.dispose();timings?.clear();
  params.coldStart=coldStart;
  sparkRemaining=0;el('ignite').removeAttribute('data-active');
  pendingSteps=0;pendingBrush=null;pendingReset=true;resetAdaptiveResolution();invalidate();timer();
}
function updateEngineLabel() {
  timings?.dispose();
  timings=new FrameTimings(isGPU?gl:null);
  el('timing-method').textContent=timings.label();
  el('adaptive-control').hidden=!isGPU;
  const adaptiveLabels={original:'Adaptive output resolution',revised:'Adaptive volume resolution',enhanced:'Adaptive volume sampling'};
  el('adaptive-label').textContent=adaptiveLabels[el('model').value];
  el('backend').textContent=isGPU?'WebGL 2':cpuVolume?'CPU volume':'Canvas 2D';
  el('frame-rate-unit').textContent=isGPU?'SUBMIT/S':'DRAW/S';
  el('solver-backend').value=currentBackend();
  el('grid-size').textContent=spatial()?engine.dims.join(' × '):engine.dims.slice(0,2).join(' × ');
  el('status').textContent=spatial()?'3D reactive flow':'2D compatibility solver';
  stage.querySelector('.orientation span').textContent=spatial()?'Perspective':'Planar slice';
  canvas.setAttribute('aria-label',spatial()?'Interactive three-dimensional fire simulation':'Interactive two-dimensional fire simulation');
  updateViews();
  updateHint();
}
function updateHint() {
  const spans=stage.querySelectorAll('.desktop-hint, .touch-hint');
  for(const s of spans) {
    const touch=s.classList.contains('touch-hint');
    s.textContent=brushMode?(running?'Drag to inject heat':'Click to inject one step'):spatial()?('Drag to orbit  ·  '+(touch?'Pinch':'Scroll')+' to zoom'):((touch?'Pinch':'Scroll')+' to zoom  ·  Tap the flame tool to ignite');
  }
  canvas.style.cursor=brushMode?'crosshair':!spatial()?'default':dragging?'grabbing':'grab';
  el('brush').setAttribute('aria-pressed',String(brushMode));
}
function replaceCanvas() {
  const prior=canvas;
  canvas=prior.cloneNode(false);
  prior.replaceWith(canvas);
}
function beginFallback(reason,volume=false) {
  const old=gl;
  const lost=old?.isContextLost();
  if(old && !lost)for(const fence of gpuFences)old.deleteSync(fence);
  gpuFences.length=0;gpuBlockedSince=null;
  if(!lost){try{renderer?.destroy();engine?.destroy();}catch(e){console.warn(e);}}
  engine=null;
  replaceCanvas();
  isGPU=false;cpuVolume=volume && selectedModel===models.enhanced;
  brushMode=!cpuVolume;gl=null;renderer=null;
  if(old && !old.isContextLost())old.getExtension('WEBGL_lose_context')?.loseContext();
  engine=createCpuEngine();
  activeModel=selectedModel;gridQuality=opts.quality;
  engine.reset(params);stateRevision++;
  updateEngineLabel();invalidate();
  if(reason) toast(reason);
}
function recoverGpu(message) {
  if(activeModel===models.enhanced){
    pauseAfterEngineFailure('WebGL');
    return;
  }
  pendingCapture=false;
  beginFallback(message);
  queueRestart();
}
function pauseAfterEngineFailure(engineName) {
  const interruptedAt=time.toFixed(2);
  pendingCapture=false;
  beginFallback();
  queueRestart();
  // A different solver can't continue the former backend's field state.
  pendingReset=false;
  solverInterrupted=true;
  setPaused(true);
  el('status').textContent=engineName+' interrupted at '+interruptedAt+' s · 2D is ready';
  toast('Play or replay to start the scene in 2D');
}
function initialize(startup=true) {
  select.value=opts.quality;
  if(startup)el('model').value='enhanced';
  el('expansion').checked=params.expansion;
  el('stable-sampling').checked=opts.stableSampling;
  el('shared-lighting').checked=opts.sharedLighting;
  try {
    gl=canvas.getContext('webgl2',{alpha:false,antialias:false,depth:false,stencil:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});
    const canRenderVolume=gl && gl.getExtension('EXT_color_buffer_float');
    el('volume-backend').disabled=!canRenderVolume;
    el('volume-backend').textContent=canRenderVolume?'3D · WebGL':'WebGL unavailable';
    if(!canRenderVolume)throw new Error(gl?'WebGL 2 floating-point render targets unavailable':'WebGL 2 context unavailable');
    isGPU=true;cpuVolume=false;brushMode=false;
    activeModel=selectedModel;gridQuality=opts.quality;
    renderer=new selectedModel.Renderer(gl);
    engine=new selectedModel.Fluid(gl,opts.quality);
    engine.reset(params);stateRevision++;
    updateEngineLabel();
    const target=canvas;
    target.addEventListener('webglcontextlost',ev=>{
      ev.preventDefault();
      if(canvas!==target) return;
      recoverGpu('Graphics context lost · restarted in 2D');
    });
  } catch(e) {
    console.warn('3D initialization:',e);
    beginFallback('Using the 2D compatibility solver');
  }
  skipTimingUntil=performance.now()+(presets[params.preset].oneShot?0:2400);
}
function sizeCanvas(scale) {
  const w=Math.max(1,stage.clientWidth),h=Math.max(1,stage.clientHeight);
  const pixels=isGPU?({low:260000,medium:520000,high:1200000,ultra:2304000}[opts.quality]):680000;
  const ratio=Math.min(devicePixelRatio,Math.sqrt(pixels/(w*h)))*scale;
  const nw=Math.max(2,Math.round(w*ratio)),nh=Math.max(2,Math.round(h*ratio));
  if(nw!==canvas.width || nh!==canvas.height) {
    canvas.width=nw;canvas.height=nh;dirty=true;
  }
}
const axes=Array.from(stage.querySelectorAll('.orientation svg path'));
let axisYaw,axisPitch,axisSpatial;
const endpoints=Array.from(stage.querySelectorAll('.orientation svg circle')).slice(1);
function updateAxes() {
  if(axisYaw===opts.yaw && axisPitch===opts.pitch && axisSpatial===spatial())return;
  axisYaw=opts.yaw;axisPitch=opts.pitch;axisSpatial=spatial();
  const sy=Math.sin(opts.yaw),cy=Math.cos(opts.yaw),sp=Math.sin(opts.pitch),cp=Math.cos(opts.pitch);
  const lines=spatial()?[[0,cp],[cy,-sy*sp],[-sy,-cy*sp]]:[[0,1],[1,0],[0,0]];
  lines.forEach(([dx,dy],i)=>{
    const x=17+16*dx,y=20-16*dy;
    axes[i].setAttribute('d','M17 20L'+x.toFixed(1)+' '+y.toFixed(1));
    endpoints[i].setAttribute('cx',x);endpoints[i].setAttribute('cy',y);
  });
}
function draw(profilePass) {
  const scale=running && !pendingCapture && opts.mode==='beauty'?renderScale:1;
  sizeCanvas(isGPU && activeModel===models.original?scale:1);
  updateAxes();
  const options={...opts,profilePass,preset:params.preset,time,stateRevision,renderScale:scale,exposure:isGPU?Math.log2(opts.exposure):opts.exposure};
  if(isGPU) renderer.draw(engine,options); else engine.draw(options);
  dirty=false;
}
function advance(dt=timestep()) {
  const brush=pendingBrush || params.brush || (sparkRemaining>0?sparkPositions[params.preset]:null);
  engine.step(dt,brush!==params.brush?{...params,brush}:params,time);
  pendingBrush=null;
  if(sparkRemaining>0){
    sparkRemaining=Math.max(0,sparkRemaining-dt);
    if(sparkRemaining===0)el('ignite').removeAttribute('data-active');
  }
  stepCount++;stateRevision++;
  if(running)simulatedSeconds+=dt;
  time+=dt;invalidate();
}
function resetIfNeeded() {
  if(!pendingReset) return;
  const modelChanged=activeModel!==selectedModel;
  const gridChanged=spatial() && opts.quality!==gridQuality;
  if(modelChanged || gridChanged){
    // A reset is serviced after all outstanding GPU fences have cleared.
    engine.destroy();engine=null;
    if(modelChanged && isGPU){
      renderer.destroy();renderer=null;
      renderer=new selectedModel.Renderer(gl);
    }
    if(cpuVolume && selectedModel!==models.enhanced){cpuVolume=false;brushMode=true;}
    engine=isGPU?new selectedModel.Fluid(gl,opts.quality):createCpuEngine();
    activeModel=selectedModel;gridQuality=opts.quality;
    updateEngineLabel();
  }
  engine.reset(params);stateRevision++;pendingReset=false;
  if(solverInterrupted){
    solverInterrupted=false;
    el('status').textContent=spatial()?'3D reactive flow':'2D compatibility solver';
    syncPlayback();
  }
}
function performanceChart(fps) {
  recentFrames.push(fps);
  if(recentFrames.length>24)recentFrames.shift();
  const c=el('frame-chart');if(!c)return;
  const ctx=c.getContext('2d'),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);ctx.beginPath();
  const high=Math.max(60,...recentFrames);
  recentFrames.forEach((v,i)=>{const x=w-(recentFrames.length-i-1)*w/23,y=h-3-(h-6)*Math.min(v,high)/high;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
  ctx.lineWidth=1.5;ctx.strokeStyle='#99a987';ctx.stroke();
}
// Prefer the display clock. A timer also drives visible embeds whose
// animation callbacks are suppressed. Only the current request can run.
let scheduledFrame=null,inFrame=false,throttledDisplay=false;
function cancelFrame() {
  if(!scheduledFrame)return;
  cancelAnimationFrame(scheduledFrame.raf);
  clearTimeout(scheduledFrame.deadline);
  scheduledFrame=null;
}
function scheduleFrame() {
  const needsFrame=running || dirty || pendingSteps || pendingCapture ||
    timings?.pending.length || gpuFences.length;
  if(inFrame || scheduledFrame || document.hidden || activePreview?.open || !needsFrame)return;
  const request={raf:0,deadline:0};
  scheduledFrame=request;
  request.raf=requestAnimationFrame(now=>deliverFrame(request,now,false));
  const stepInterval=Math.max(16,1000*timestep()/speed);
  const delay=throttledDisplay && running?
    Math.max(4,stepInterval-(performance.now()-last)):100;
  request.deadline=setTimeout(()=>deliverFrame(request,performance.now(),true),delay);
}
function invalidate() {
  dirty=true;
  scheduleFrame();
}
function deliverFrame(request,now,throttled) {
  if(scheduledFrame!==request)return;
  cancelFrame();
  throttledDisplay=throttled;
  inFrame=true;
  try {frame(now);}
  finally {inFrame=false;scheduleFrame();}
}
function availableGpuSlot(now) {
  if(gl.isContextLost())throw new Error('WebGL context lost');
  while(gpuFences.length){
    const fence=gpuFences[0];
    const result=gl.clientWaitSync(fence,0,0);
    if(result===gl.TIMEOUT_EXPIRED){
      // No GPU work has completed while visible. A failed fence must
      // never block the simulation, capture and restart controls forever.
      gpuBlockedSince??=now;
      if(now-gpuBlockedSince>=gpuStallLimit)throw new Error('WebGL stopped completing frames');
      break;
    }
    if(result===gl.WAIT_FAILED)throw new Error('WebGL completion check failed');
    gl.deleteSync(fence);gpuFences.shift();
    gpuBlockedSince=null;
  }
  const overloaded=activeModel===models.enhanced && renderScale<=.84;
  const limit=running && !overloaded && !dragging && !pendingReset && !pendingCapture?2:1;
  if(gpuFences.length<limit)return true;
  blockedFrames++;
  return false;
}
function saveFrame() {
  pendingCapture=false;
  const name='ignis-'+params.preset+'-'+time.toFixed(2)+'s.png';
  canvas.toBlob(blob=>{
    if(!blob){toast('Image capture unavailable');return;}
    const link=document.createElement('a'),url=URL.createObjectURL(blob);
    link.href=url;link.download=name;link.click();
    setTimeout(()=>URL.revokeObjectURL(url),30000);toast('Viewport image saved');
  },'image/png');
}
// Without GPU timers a slow solver can make every shade-rate reduction look
// necessary. Trial one smaller raster and restore it when cadence fails to improve.
function adjustResolution(fps,now) {
  if(!isGPU || opts.mode!=='beauty' || !autoResolution.checked || dragging ||
    now<skipTimingUntil || elapsed<.35)return;
  const target=Math.min(60,speed/timestep());
  const {solver,render}=timings.values;
  const measured=Boolean(timings.ext && solver!==null && render!==null);
  if(resolutionTrial) {
    resolutionTrial.frames+=frameCount;
    resolutionTrial.seconds+=elapsed;
    if(resolutionTrial.seconds<2)return;
    const improved=resolutionTrial.frames/resolutionTrial.seconds>resolutionTrial.fps*1.04;
    if(!improved) {
      renderScale=resolutionTrial.previous;
      skipTimingUntil=now+20000;
      invalidate();
    }
    resolutionTrial=null;slowWindows=0;fastWindows=0;
    return;
  }
  const enhanced=activeModel===models.enhanced;
  const constrained=measured?render>solver*.55:(blockedFrames>=2 || (enhanced && fps<target*.70));
  slowWindows=fps<target*.87 && constrained?slowWindows+1:0;
  fastWindows=fps>target*.97 && (!measured || solver+render<.8*1000/target)?fastWindows+1:0;
  const adaptiveScales=enhanced?enhancedScales:legacyScales;
  const rung=adaptiveScales.indexOf(renderScale);
  let next=rung;
  if(slowWindows>=(enhanced?2:3))next=Math.min(rung+1,adaptiveScales.length-1);
  else if(fastWindows>=12)next=Math.max(rung-1,0);
  if(next===rung)return;
  if(next>rung && !measured)resolutionTrial={previous:renderScale,fps,frames:0,seconds:0};
  renderScale=adaptiveScales[next];slowWindows=0;fastWindows=0;
  skipTimingUntil=now+1600;invalidate();
}
function updateStatistics(now,profiling) {
  if(profiling) {
    for(const key of ['solver','render']){
      const ms=timings.values[key];el(key+'-ms').textContent=ms===null?'—':ms.toFixed(2)+' ms';
    }
    for(const key of ['solver','render',...renderPasses]){
      const ms=timings.percentile95(key);
      el(key+'-p95').textContent=ms===null?'—':ms.toFixed(2)+' ms';
    }
    el('draw-size').textContent=canvas.width+' × '+canvas.height;
    const volumeRenderer=isGPU?renderer:(cpuVolume?engine:null);
    el('volume-size').textContent=volumeRenderer?(volumeRenderer.renderWidth??canvas.width)+' × '+(volumeRenderer.renderHeight??canvas.height):'—';
    el('ray-samples').textContent=isGPU && renderer.raySamples?String(renderer.raySamples):'—';
    el('step-rate').textContent=running && elapsed>0?Math.round(stepCount/elapsed)+' Hz measured':'Paused';
    el('sim-rate').textContent=running && elapsed>0?(simulatedSeconds/elapsed).toFixed(2)+'×':'—';
    el('clock-drop').textContent=droppedClockTime.toFixed(2)+' s';
    const sorted=frameGaps.slice().sort((a,b)=>a-b);
    el('frame-gap').textContent=sorted.length?Math.round(sorted[Math.floor((sorted.length-1)*.95)])+' ms':'—';
  }
  if(elapsed>0 && running) {
    const fps=frameCount/Math.max(.001,elapsed);
    el('fps').textContent=String(Math.round(fps));
    const simRate=simulatedSeconds/elapsed;
    el('live').textContent=activeModel===models.enhanced && elapsed>=.35 && simRate<speed*.88
      ? 'Behind · '+simRate.toFixed(2)+'×' : 'Live';
    performanceChart(fps);
    adjustResolution(fps,now);
  }
  frameCount=0;stepCount=0;simulatedSeconds=0;blockedFrames=0;elapsed=0;lastStats=now;
}
function finishMeasurement(now) {
  const sample=playbackMeasurement;
  if(!sample)return;
  playbackMeasurement=null;
  el('measure-run').textContent='Measure 10 s';
  const result=el('measure-result');
  result.hidden=false;
  if(now===undefined){
    result.textContent='Measurement canceled';
    return;
  }
  const count=sample.gaps.length;
  const duration=(now-sample.start)/1000;
  const sorted=sample.gaps.sort((a,b)=>a-b);
  const gap=count?sorted[Math.ceil(.95*count)-1].toFixed(1)+' ms':'—';
  const rate=((time-sample.simStart)/duration).toFixed(2);
  result.textContent=sample.label+' · '+(sample.frames/duration).toFixed(1)+' submissions/s · P95 gap '+gap+' · '+rate+'× simulation';
}
function frame(now) {
  if(document.hidden || activePreview?.open)return;
  const realDelta=last?Math.max(0,now-last)/1000:0;
  last=now;
  try {
    const profiling=el('profiler').open;
    if(timings.pending.length)timings.poll();
    if(running){
      const demand=accumulator+realDelta*speed;
      const enhanced=activeModel===models.enhanced;
      const debtLimit=enhanced?enhancedDebtLimit:timestep()*(speed>1?3:2);
      accumulator=Math.min(demand,debtLimit);
      if(enhanced)droppedClockTime+=demand-accumulator;
      elapsed+=realDelta;
    }
    if(!isGPU || availableGpuSlot(now)){
      resetIfNeeded();
      const dt=timestep();
      const measure=profiling || (isGPU && autoResolution.checked && renderSerial%6===0 && timings.ext);
      const manual=pendingSteps>0;
      let ticks=0;
      if(manual){
        pendingSteps--;ticks=1;
      }else if(running){
        ticks=Math.floor((accumulator+1e-7)/dt);
        // Bound work per presentation without changing the physics step.
        if(activeModel===models.enhanced){
          const maxTicks=isGPU && (opts.quality==='high' || opts.quality==='ultra')?2:3;
          ticks=Math.min(maxTicks,ticks);
        }
      }
      if(ticks){
        const solve=()=>{for(let i=0;i<ticks;i++)advance(dt);};
        if(measure)timings.measure('solver',solve);
        else solve();
        if(manual)timer();
        else accumulator=Math.max(0,accumulator-ticks*dt);
      }
      if(dirty || pendingCapture){
        if(profiling && isGPU && activeModel===models.enhanced && renderSerial%2){
          const pass=renderPasses[Math.floor(renderSerial/2)%renderPasses.length];
          timings.measurePass(pass,draw);
        }else if(measure)timings.measure('render',draw);
        else draw();
        renderSerial++;
        if(running){
          frameCount++;
          if(playbackMeasurement && now>=playbackMeasurement.start){
            const sample=playbackMeasurement;
            if(sample.last)sample.gaps.push(now-sample.last);
            sample.last=now;
            sample.frames++;
          }
          if(lastPresentation){
            frameGaps.push(now-lastPresentation);
            if(frameGaps.length>42)frameGaps.shift();
          }
          lastPresentation=now;
        }
        if(pendingCapture)saveFrame();
        if(isGPU){
          const fence=gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE,0);
          if(fence)gpuFences.push(fence);
          gl.flush();
        }
      }
    }
    if(playbackMeasurement && now-playbackMeasurement.start>=10000)finishMeasurement(now);
    if(now-lastStats>600 || !running && profiling && !timings.pending.length)updateStatistics(now,profiling);
    if(running && Math.floor(now/80)!==Math.floor((now-realDelta*1000)/80))timer();
  } catch(e) {
    console.error('Simulation runtime:',e);
    if(isGPU){
      recoverGpu('Graphics interrupted · restarted in 2D');
    }else if(cpuVolume){
      pauseAfterEngineFailure('CPU volume');
    }else{
      setPaused(true);
      solverInterrupted=true;dirty=false;pendingCapture=false;pendingSteps=0;
      el('status').textContent='Solver interrupted';
      toast('Restart the simulation to retry');
    }
  }
}
function choosePreset(key) {
  const preset=presets[key];
  if(!preset)return;
  sceneSettings[params.preset]=Object.fromEntries(controls.filter(k=>k!=='exposure').map(k=>[k,params[k]]));
  const {yaw,pitch,distance,targetHeight}=opts;
  sceneCameras[params.preset]={yaw,pitch,distance,targetHeight};
  const {fuel,wind,turbulence,oxygen}=preset;
  Object.assign(params,{fuel,wind,turbulence,oxygen},sceneSettings[key]||{},{preset:key});
  Object.assign(opts,sceneCameras[key]??defaultSceneCamera());
  if(preset.enhanced && selectedModel!==models.enhanced){
    selectedModel=models.enhanced;
    el('model').value='enhanced';
    toast('This scenario uses the Enhanced model');
  }
  el('scene-title').textContent=preset.name;
  el('scene-id').textContent=preset.oneShot?'ONE SHOT':'REACTIVE FLOW';
  for(const btn of document.querySelectorAll('[data-preset]')) {
    const chosen=btn.dataset.preset===key;btn.classList.toggle('selected',chosen);btn.setAttribute('aria-pressed',String(chosen));
  }
  syncControls();updateViews();queueRestart(false);
}
document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>choosePreset(b.dataset.preset)));
for(const name of controls)el(name).addEventListener('input',e=>{
  if(name==='exposure')opts.exposure=Number(e.target.value);else params[name]=Number(e.target.value);
  syncControls();if(name==='exposure')invalidate();
});
const legendLabels={temperature:['Gas temperature','Cool','Hot'],wood:['Wood temperature','Cool','Hot'],smoke:['Soot density','Clear','Dense'],velocity:['Flow speed','Slow','Fast']};
function updateViews(){
  const woodView=el('wood-view');
  const revised=selectedModel===models.revised;
  const enhanced=selectedModel===models.enhanced;
  const preset=presets[params.preset];
  el('replay-event').hidden=!preset.oneShot;
  el('legacy-original').disabled=el('legacy-revised').disabled=Boolean(preset.enhanced);
  el('enhanced-settings').hidden=!enhanced;
  el('cpu-volume-backend').disabled=!enhanced;
  el('pass-timings').hidden=!enhanced || !isGPU;
  woodView.disabled=!revised || !isGPU || !isWoodScene();
  el('ignition-controls').hidden=!revised;
  if(woodView.disabled && opts.mode==='wood')opts.mode='beauty';
  el('mode').value=opts.mode;
  invalidate();
  const legend=el('field-legend'), labels=legendLabels[opts.mode];
  legend.hidden=!labels;
  if(labels){
    legend.dataset.mode=opts.mode;
    legend.querySelectorAll('span').forEach((span,i)=>span.textContent=labels[i]);
  }
}
el('mode').addEventListener('change',e=>{opts.mode=e.target.value;updateViews();});
el('model').addEventListener('change',event=>{
  const next=models[event.target.value];
  if(!next || next===selectedModel)return;
  if(presets[params.preset].enhanced){
    event.target.value='enhanced';
    return;
  }
  selectedModel=next;
  syncControls();updateViews();queueRestart(false);
  toast('Model changed · scene restarted');
});
for(const [id,option] of [['stable-sampling','stableSampling'],['shared-lighting','sharedLighting']]){
  el(id).addEventListener('change',event=>{
    opts[option]=event.target.checked;
    timings.dispose();timings.clear();resetAdaptiveResolution();invalidate();
  });
}
el('expansion').addEventListener('change',event=>{
  params.expansion=event.target.checked;queueRestart(false);
  toast('Expansion changed · scene restarted');
});
select.addEventListener('change',()=>{
  opts.quality=select.value;
  if(spatial())queueRestart();
  else {resetAdaptiveResolution();invalidate();}
});
el('speed').addEventListener('change',e=>{finishMeasurement();speed=Number(e.target.value);resetAdaptiveResolution();invalidate();});
el('solver-backend').addEventListener('change',event=>{
  const requested=event.target.value;
  if(requested===currentBackend())return;
  if(requested==='cpu-volume' && selectedModel!==models.enhanced){
    event.target.value=currentBackend();return;
  }
  queueRestart();
  if(requested==='gpu'){
    engine.destroy();engine=null;
    replaceCanvas();
    initialize(false);
  }else{
    try{beginFallback(undefined,requested==='cpu-volume');}
    catch(error){
      console.warn('CPU volume initialization:',error);
      beginFallback('CPU volume unavailable · using 2D');
    }
  }
  pendingReset=false;solverInterrupted=false;
  resetPlaybackClock();syncPlayback();invalidate();
  if(currentBackend()===requested)toast('Engine changed · scene restarted');
});
function togglePlayback() {
  if(solverInterrupted)queueRestart();
  setPaused(running);
}
el('pause').onclick=togglePlayback;
el('restart').onclick=()=>{
  queueRestart(false);
  if(solverInterrupted)setPaused(false);
  toast('Simulation restarted');
};
el('replay-event').onclick=()=>{
  if(!presets[params.preset].oneShot)return;
  queueRestart(false);
  setPaused(false);
};
el('cold-start').onclick=()=>{queueRestart(true);setPaused(false);toast('Cold start · use Ignite base to add heat');};
el('ignite').onclick=()=>{
  sparkRemaining=.95;el('ignite').setAttribute('data-active','');
  if(!running)setPaused(false);
  toast('Applying heat at the fuel bed');
};
el('step').onclick=()=>{
  if(solverInterrupted)queueRestart();
  setPaused(true);pendingSteps++;invalidate();
};
el('reset-camera').onclick=()=>{Object.assign(opts,defaultSceneCamera());invalidate();};
el('brush').onclick=()=>{brushMode=!brushMode;params.brush=null;updateHint();};
el('fullscreen').onclick=async()=>{
  try {
    if(document.fullscreenElement)await document.exitFullscreen();
    else if(stage.requestFullscreen)await stage.requestFullscreen();
    else stage.classList.toggle('expanded-stage');
  }catch{stage.classList.toggle('expanded-stage');}
  invalidate();
};
el('capture').onclick=()=>{pendingCapture=true;invalidate();};
autoResolution.addEventListener('change',()=>{
  resetAdaptiveResolution();invalidate();
});
el('profiler').addEventListener('toggle',()=>{
  if(el('profiler').open){updateStatistics(performance.now(),true);scheduleFrame();}
});
el('measure-run').onclick=()=>{
  if(playbackMeasurement){finishMeasurement();return;}
  const result=el('measure-result');
  result.hidden=false;
  if(!running){result.textContent='Play the simulation to measure a run';return;}
  const label=[el('backend').textContent,presets[params.preset].name,opts.quality].join(' · ');
  playbackMeasurement={start:performance.now(),simStart:time,label,last:0,frames:0,gaps:[]};
  el('measure-run').textContent='Cancel measurement';
  result.textContent='Measuring 10 seconds…';
};
function brushAt(e) {
  const r=stage.getBoundingClientRect(),nx=(e.clientX-r.left)/r.width,ny=(e.clientY-r.top)/r.height;
  if(cpuVolume)return engine.brushAt(nx,ny,opts);
  if(!isGPU)return engine.brushAt(nx*canvas.width,ny*canvas.height,opts);
  const ray=cameraRay(nx*2-1,1-ny*2,r.width/r.height,opts);
  const sx=Math.sin(opts.yaw),sz=Math.cos(opts.yaw),o=ray.origin,d=ray.direction;
  const t=-(sx*o[0]+sz*o[2])/(sx*d[0]+sz*d[2]);
  const world=o.map((v,i)=>v+d[i]*t);
  return [clamp(world[0]/2+.5,.03,.97),clamp(world[1]/2.4,.035,.96),clamp(world[2]/2+.5,.03,.97)];
}
function pinchDistance(){const [a,b]=activePointers.values();return Math.max(1,Math.hypot(a.x-b.x,a.y-b.y));}
stage.addEventListener('pointerdown',e=>{
  if(e.target!==canvas || e.button>0)return;
  stage.setPointerCapture(e.pointerId);
  activePointers.set(e.pointerId,{x:e.clientX,y:e.clientY});
  dragging=true;activeBrush=(brushMode||e.shiftKey)&&activePointers.size===1;
  params.brush=activeBrush?brushAt(e):null;
  if(activeBrush && !running){pendingBrush=params.brush;pendingSteps++;invalidate();}
  updateHint();
});
stage.addEventListener('pointermove',e=>{
  const p=activePointers.get(e.pointerId);if(!p)return;
  const dx=e.clientX-p.x,dy=e.clientY-p.y,oldGap=activePointers.size===2?pinchDistance():0;
  p.x=e.clientX;p.y=e.clientY;
  if(oldGap)opts.distance=clamp(opts.distance*oldGap/pinchDistance(),1.75,6.8);
  else if(activeBrush)params.brush=brushAt(e);
  else if(spatial()){opts.yaw-=dx*.007;opts.pitch=clamp(opts.pitch+dy*.006,-.06,1.25);}
  invalidate();
});
function endPointer(e){activePointers.delete(e.pointerId);dragging=activePointers.size>0;params.brush=null;activeBrush=false;updateHint();}
stage.addEventListener('pointerup',endPointer);stage.addEventListener('pointercancel',endPointer);stage.addEventListener('lostpointercapture',endPointer);
stage.addEventListener('wheel',e=>{
  if(e.target!==canvas)return;e.preventDefault();opts.distance=clamp(opts.distance*Math.exp(clamp(e.deltaY,-200,200)*.0012),1.75,6.8);invalidate();
},{passive:false});
document.addEventListener('click',e=>{if(!e.target.closest('#profiler'))el('profiler').open=false;});
document.addEventListener('keydown',e=>{
  if(document.querySelector('dialog[open]'))return;
  if(e.ctrlKey||e.metaKey||e.altKey||e.target.matches('select,input,textarea,button')||e.repeat)return;
  if(e.code==='Space'){e.preventDefault();togglePlayback();}
  if(e.key.toLowerCase()==='b')el('brush').click();
  if(e.key.toLowerCase()==='r')el('restart').click();
  if(e.key==='.')el('step').click();
  if(e.key==='Escape')stage.classList.remove('expanded-stage');
});
const video=el('proof-video');
function linkDialog(dialogId,triggerIds,closeId,onOpen,onClose) {
  const dialog=el(dialogId);
  const open=triggerId=>{
    if(dialog.open)return;
    try {dialog.showModal();}
    catch(error){
      console.warn('Preview could not open:',error);
      toast('Preview could not open');
      return;
    }
    activePreview=dialog;
    resetPlaybackClock();
    syncPlayback();
    onOpen(triggerId);
  };
  for(const triggerId of [].concat(triggerIds))el(triggerId).onclick=()=>open(triggerId);
  el(closeId).onclick=()=>dialog.close();
  dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close();});
  dialog.addEventListener('close',()=>{
    // A close event can arrive after the same dialog has reopened.
    if(dialog.open)return;
    onClose?.();
    if(activePreview!==dialog)return;
    activePreview=null;
    resetPlaybackClock();
    syncPlayback();
    invalidate();
  });
}
linkDialog('capture-dialog',['watch-capture','watch-presets'],'close-dialog',triggerId=>{
  const original=selectedModel===models.original, enhanced=selectedModel===models.enhanced;
  const reel=triggerId==='watch-presets';
  const src=reel?'./media/presets-reel.mp4':enhanced?'./media/wood-enhanced.mp4':original?'./media/wood-crib.mp4':'./media/wood-revised.mp4';
  if(video.getAttribute('src')!==src)video.setAttribute('src',src);
  video.poster=reel?'./media/presets-reel-poster.webp':enhanced?'./media/wood-enhanced.png':original?'./media/wood-crib.png':'./media/wood-revised.png';
  const largeOriginal=original && !reel;
  video.width=largeOriginal?840:640;video.height=reel?548:largeOriginal?600:460;
  el('record-title').textContent=reel?'Scenario recordings':'Wood crib · '+(enhanced?'Enhanced':original?'Original':'Revised');
  el('record-details').textContent=reel?'WebGL recordings of four live simulations':'WebGL solver recording · 48 × 72 × 48 cells';
  video.play().catch(()=>{});
},()=>video.pause());
const comparisonBackend=el('comparison-backend');
function showComparison() {
  const planar=comparisonBackend.value==='cpu';
  const enhanced=el('comparison-model').value==='enhanced';
  const label=enhanced?'Enhanced':'Revised';
  el('comparison-film').hidden=planar || !enhanced;
  el('comparison-before').src=planar?'./media/wood-planar-after.webp':'./media/wood-after.webp';
  el('comparison-after').src=enhanced
    ?(planar?'./media/wood-enhanced-planar.webp':'./media/wood-enhanced.webp')
    :(planar?'./media/wood-planar-before.webp':'./media/wood-before.webp');
  el('comparison-after').alt='Wood crib at three seconds with the '+label+' model';
  el('comparison-updated-label').textContent=label;
  el('comparison-reveal').setAttribute('aria-valuetext',el('comparison-reveal').value+'% '+label);
}
el('comparison-model').addEventListener('change',showComparison);
comparisonBackend.addEventListener('change',showComparison);
linkDialog('comparison-dialog','show-comparison','close-comparison',()=>{
  comparisonBackend.value=spatial()?'gpu':'cpu';
  el('comparison-model').value=selectedModel===models.revised?'revised':'enhanced';
  showComparison();
});
el('comparison-reveal').addEventListener('input',event=>{
  const reveal=event.currentTarget.valueAsNumber;
  event.currentTarget.setAttribute('aria-valuetext',reveal+'% '+el('comparison-updated-label').textContent);
  el('comparison').style.setProperty('--reveal',reveal+'%');
  el('comparison-baseline-label').hidden=reveal===100;
  el('comparison-updated-label').hidden=reveal===0;
});
new ResizeObserver(invalidate).observe(stage);
function suspendDisplay() {
  resetPlaybackClock();
  activePointers.clear();
  dragging=false;activeBrush=false;params.brush=null;
}
function restoreDisplay() {
  resetPlaybackClock();
  updateHint();
  invalidate();
}
document.addEventListener('visibilitychange',()=>{
  if(document.hidden)suspendDisplay();
  else restoreDisplay();
});
window.addEventListener('pagehide',suspendDisplay);
window.addEventListener('pageshow',event=>{if(event.persisted)restoreDisplay();});
syncControls();
initialize();
scheduleFrame();

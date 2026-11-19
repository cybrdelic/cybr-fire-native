// Reduced 2D path. The velocity projection, transported species, heat release,
// and oxygen consumption all run in the CPU solver on the same Eulerian grid.
const NX = 104, NY = 124, N = NX * NY;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const unit = x => clamp(x, 0, 1);
const isWood = preset => preset==='wood'||preset==='crosswind';
const ramp = (lo,hi,value) => {
  const x=unit((value-lo)/(hi-lo));
  return x*x*(3-2*x);
};
const flightBound=NY-2;

const defaults = {preset: 'wood', fuel: .66, oxygen: .78, turbulence: .58, wind: 0, coldStart:false};
const burners = {
  wood: [[-.52,.059,.81],[-.353,.083,1.21],[-.128,.107,1.44],[.07,.096,1.21],[.278,.102,1.51],[.455,.055,1.04]],
  pool: [[-.53,.11,.8],[-.32,.14,1.1],[-.1,.16,.97],[.12,.14,1.16],[.34,.14,1.1],[.53,.09,.84]],
  jet: [[-.045,.06,1.2],[.035,.055,1.1]],
  crosswind: [[-.48,.09,.91],[-.29,.13,1.1],[-.12,.11,.85],[.07,.10,1.12],[.24,.08,.8]]
};
// The six entries approximate a soot continuum across the displayed thermal range.
const sootColors = [
  [1,.019,.002],[1,.059,.004],[1,.155,.013],[1,.325,.037],[1,.515,.097],[1,.699,.236]
];
function emissionLut() {
  const out = new Uint8Array(128*128*3);
  for(let i=0;i<128;i++) {
    const t=i/127*(sootColors.length-1), a=Math.min(sootColors.length-2,t|0), f=t-a;
    for(let j=0;j<128;j++) {
      const energy=j/127*3.8;
      for(let k=0;k<3;k++) {
        const channel=energy*(sootColors[a][k]+(sootColors[a+1][k]-sootColors[a][k])*f);
        const film=channel*(2.18*channel+.048)/(channel*(2.12*channel+.61)+.145);
        out[(i*128+j)*3+k]=255*Math.sqrt(unit(film));
      }
    }
  }
  return out;
}
// Coherent structure no smaller than a few cells, seeded once and carried in
// the fluid as a scalar. A step never re-samples rapidly stretched coordinates.
function seedFilaments() {
  const out=new Float32Array(N), random=randomSequence(5869521);
  for(const spacing of [5,10,16]) {
    const w=Math.ceil((NX+NY*.42)/spacing)+1, h=Math.ceil((NY+NX*.26)/spacing)+1;
    const nodes=Float32Array.from({length:w*h},()=>2*random()-1);
    for(let y=0;y<NY;y++) {
      for(let x=0;x<NX;x++) {
        const sx=(x+.42*y)/spacing, sy=(y+.26*(NX-1-x))/spacing;
        const i=sx|0, j=sy|0, fx=sx-i, fy=sy-j;
        const u=fx*fx*(3-2*fx), v=fy*fy*(3-2*fy), k=j*w+i;
        const low=nodes[k]+u*(nodes[k+1]-nodes[k]);
        const high=nodes[k+w]+u*(nodes[k+w+1]-nodes[k+w]);
        out[y*NX+x]+=(low+v*(high-low))/3;
      }
    }
  }
  return out;
}
function randomSequence(seed) {
  return () => ((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
}
function state() {
  return {u:new Float32Array(N), v:new Float32Array(N), heat:new Float32Array(N),
    filament:new Float32Array(N), fuel:new Float32Array(N), smoke:new Float32Array(N), air:new Float32Array(N)};
}
function interp(a, i, x, y) {
  const bottom=a[i]+(a[i+1]-a[i])*x, top=a[i+NX]+(a[i+NX+1]-a[i+NX])*x;
  return bottom+(top-bottom)*y;
}
function corrected(original, predicted, i, departure, forward, fx, fy) {
  const q=predicted[i]+.5*(original[i]-interp(predicted,forward,fx,fy));
  const a=original[departure], b=original[departure+1], c=original[departure+NX], d=original[departure+NX+1];
  return clamp(q,Math.min(a,b,c,d),Math.max(a,b,c,d));
}
// Interpolate the middle two knots with a bounded cubic. A smooth slope
// removes the rectilinear edge from bilinear display samples of bright fronts.
function cubic(a,b,c,d,t) {
  const delta=c-b, m=.33*(c-a), n=.33*(d-b);
  const y=b+t*(m+t*((3*delta-2*m-n)+t*(m+n-2*delta)));
  return clamp(y,Math.min(b,c),Math.max(b,c));
}
function cubicFront(a,i,fx,fy) {
  const x=i%NX;
  if(x<1||x>NX-3||i<NX||i>=N-2*NX)return interp(a,i,fx,fy);
  const lo=i-NX, hi=i+NX, hh=i+2*NX;
  return cubic(
    cubic(a[lo-1],a[lo],a[lo+1],a[lo+2],fx),
    cubic(a[i-1],a[i],a[i+1],a[i+2],fx),
    cubic(a[hi-1],a[hi],a[hi+1],a[hi+2],fx),
    cubic(a[hh-1],a[hh],a[hh+1],a[hh+2],fx),
    fy);
}
function opacityAt(table,t) {
  t=Math.min(t,2047);
  const i=t|0;
  return table[i]+(t-i)*(table[i+1]-table[i]);
}
function grad(ctx, x0, y0, r, stops) {
  const g=ctx.createRadialGradient(x0,y0,0,x0,y0,r);
  for(const [t,c] of stops) g.addColorStop(t,c);
  return g;
}

export class Fallback {
  constructor(canvas) {
    this.canvas=canvas;
    this.ctx=canvas.getContext('2d',{alpha:false});
    if(!this.ctx) throw Error('A fresh 2D canvas is required for the reduced solver');
    this.dims=[NX,NY,1];
    this.slice=document.createElement('canvas');
    this.slice.width=NX; this.slice.height=NY;
    this.sliceContext=this.slice.getContext('2d');
    this.image=this.sliceContext.createImageData(NX,NY);
    this.a=state(); this.b=state(); this.curl=new Float32Array(N);
    this.advectedHeat=new Float32Array(N); this.advectedFuel=new Float32Array(N);
    this.departure=new Uint16Array(N);
    this.departureX=new Float32Array(N); this.departureY=new Float32Array(N);
    this.pressure=new Float32Array(N); this.divergence=new Float32Array(N);
    this.source=new Float32Array(N); this.burn=new Float32Array(N);
    this.wood={heat:new Float32Array(NX),mass:new Float32Array(NX),char:new Float32Array(NX),
      coverage:new Float32Array(NX),feed:new Float32Array(NX),glow:new Float32Array(NX)};
    this.ignitionHeat = new Float32Array(121);
    for(let r=0;r<this.ignitionHeat.length;r++)this.ignitionHeat[r]=(1-r/121)**2;
    this.xWave=[new Float32Array(NX),new Float32Array(NX),new Float32Array(NX)];
    this.yWave=[new Float32Array(NY),new Float32Array(NY),new Float32Array(NY)];
    this.emission=emissionLut();
    this.opacity=Float32Array.from({length:2049},(_,i)=>1-Math.exp(-i/256));
    this.filamentSeed=seedFilaments();
    this.luminance=new Float32Array(N);this.sheets=new Float32Array(N);
    this.sceneCache=null;this.revision=0;this.imageRevision=-1;
    this.sparks=[];
    this.reset();
  }
  reset(params={}) {
    this.params={...defaults,...params};
    const kind=this.params.preset;
    const nozzle=kind==='jet';
    const burning=this.params.coldStart?0:unit(this.params.fuel)*unit(this.params.oxygen*2);
    const wood=isWood(kind);
    const sources=burners[kind]||burners.wood;
    this.random=randomSequence(938102);
    this.lightX=0;this.radiantPower=burning; this.sparks.length=0; this.time=0; this.sparkBudget=0;this.revision++;
    this.burn.fill(0); this.pressure.fill(0); this.curl.fill(0);
    this.wood.coverage.fill(0);this.wood.feed.fill(0);this.wood.glow.fill(0);
    this.wood.heat.fill(0);this.wood.mass.fill(1);this.wood.char.fill(0);
    this.emberPower=0;
    for(const value of Object.values(this.b)) value.fill(0);
    for(let y=0;y<NY;y++) for(let x=0;x<NX;x++) {
      const i=x+y*NX, h=y/(NY-1), xx=2*x/(NX-1)-1;
      const y0=nozzle?.08:.035;
      const rise=Math.max(0,h-y0);
      let src=0, flame=0;
      const drift=this.params.wind*rise*rise*.85;
      for(let k=0;k<sources.length;k++) {
        const [center,width,power]=sources[k];
        const dx=(xx-center)/width;
        src+=power*Math.exp(-dx*dx-((h-y0)/(nozzle?.024:.021))**2);
        const tongueHeight=(nozzle?.67:.48)+.12*Math.sin(k*8.1)+.09*Math.cos(k*4.9);
        const wavering=(.024+.09*rise)*(Math.sin(22*rise+k*4.4)+.37*Math.sin(43*rise+k*1.41));
        const off=(xx-center*(1-.8*rise)-drift-wavering)/(width*(1.25-.65*rise));
        flame+=Math.exp(-off*off*1.05)*Math.pow(unit(1-rise/tongueHeight),.82)*power;
      }
      const localPattern=this.filamentSeed[i];
      const ripple=.84+.27*localPattern;
      const plume=Math.exp(-1*((xx-drift-.075*Math.sin(15*h))/(.26+.20*h))**2);
      // Finite source openings follow the log coverage without a spatial comb.
      const porosity=nozzle?1:.73+.31*this.filamentSeed[4*NX+x];
      this.source[i]=src<.00045?0:Math.min(1.8,src)*porosity;
      this.a.heat[i]=Math.min(1.65,flame*1.45*ripple)*clamp(1.61*burning,0,1.55);
      this.a.fuel[i]=Math.min(.7,flame*.48+src*.22)*burning;
      this.a.smoke[i]=burning*(.12+.56*h)*plume*Math.sin(Math.PI*h)**.5;
      this.a.filament[i]=this.filamentSeed[i];
      this.a.air[i]=this.params.oxygen*(1-Math.min(.53,burning*flame*.31));
      this.a.u[i]=7*burning*Math.sin(h*11)*plume+this.params.wind*13;
      this.a.v[i]=burning*(nozzle?55:38)*plume*Math.max(.05,(1-h*.8));
      this.burn[i]=Math.min(1,flame)*burning;
      if(wood&&y===4) {
        const bed=this.wood, coverage=unit(src*1.25);
        bed.coverage[x]=coverage;
        bed.heat[x]=1.07*burning*Math.sqrt(coverage);
        bed.char[x]=.18*burning*coverage;
      }
    }
    for(let i=0;i<Math.round(burning*34);i++) this.spawnSpark(this.random()*.72);
  }
  advanceWood(dt,oxygen,bx,by) {
    const bed=this.wood, gas=this.a.heat, supply=unit(this.params.fuel);
    let heatBudget=0, fuelBudget=0;
    for(let x=2;x<NX-2;x++) {
      const coverage=bed.coverage[x];
      if(coverage<.002)continue;
      const heat=bed.heat[x];
      const airHeat=.32*gas[x+4*NX]+.40*gas[x+7*NX]+.28*gas[x+10*NX];
      const previous=x>2?bed.heat[x-1]:heat, next=x<NX-3?bed.heat[x+1]:heat;
      const brushDistance=(x-bx)**2+(4.7-by)**2;
      const stimulus=brushDistance<165 ? 5.2*(1-brushDistance/165)**2 : 0;
      const pyrolysis=.095*supply*bed.mass[x]*coverage*ramp(.16,.77,heat);
      const charBurn=.066*oxygen*bed.char[x]*ramp(.18,.87,heat);
      bed.mass[x]=Math.max(0,bed.mass[x]-dt*pyrolysis);
      bed.char[x]=unit(bed.char[x]+dt*(.43*pyrolysis-charBurn));
      const conductive=.72*(previous+next-2*heat);
      const convective=2.75*airHeat*(1-.46*unit(heat));
      bed.heat[x]=clamp(heat+dt*(stimulus+conductive+convective+4.6*charBurn-(.43+.29*heat*heat)*heat-1.4*pyrolysis),0,2.1);
      bed.feed[x]=17.5*pyrolysis;
      bed.glow[x]=ramp(.18,1.25,bed.heat[x])*(.38+.62*unit(2.3*bed.char[x]));
      heatBudget+=bed.glow[x]*coverage;
      fuelBudget+=coverage;
    }
    this.emberPower=fuelBudget?heatBudget/fuelBudget:0;
  }
  spawnSpark(age=0) {
    const r=this.random, narrow=this.params.preset==='jet'?.14:.78;
    const x=NX*(.5+(r()-.5)*narrow*.55);
    this.sparks.push({x:x+Math.sin(age*7+x)*age*5,y:7+age*60,u:(r()-.5)*6,
      v:25+r()*26,size:.32+r()*.7,life:1.0+r()*1.4,age,seed:r()*Math.PI*2});
  }
  step(dt,params={}, time) {
    if(!(dt>0)) return;
    if(params.preset && params.preset!==this.params.preset) this.reset(params);
    Object.assign(this.params,params);
    dt=Math.min(.065,dt);
    const turns=Math.ceil(dt/.036), sub=dt/turns;
    for(let j=0;j<turns;j++) {this.time+=sub; this.advance(sub);}
  }
  advance(dt) {
    const {a,b,curl,divergence,pressure,burn,source,filamentSeed}=this;
    const p=this.params, t=this.time, jet=p.preset==='jet', wood=isWood(p.preset), ox=unit(p.oxygen);
    const forces=.25+1.35*p.turbulence, ambientWind=p.wind*30;
    const [x1,x2,x3]=this.xWave, [y1,y2,y3]=this.yWave;
    for(let x=0;x<NX;x++) {
      x1[x]=Math.sin(x*.11+t*1.6); x2[x]=Math.sin(x*.239-t*2.6); x3[x]=Math.cos(x*.055+t*1.13);
    }
    for(let y=0;y<NY;y++) {
      y1[y]=Math.cos(y*.132-t*2.8); y2[y]=Math.sin(y*.223-t*3.7); y3[y]=Math.sin(y*.069-t*1.85);
    }
    for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
      curl[i]=.5*(a.v[i+1]-a.v[i-1]-a.u[i+NX]+a.u[i-NX]);
    }
    // A forward semi-Lagrangian prediction lets us reverse-advect heat and
    // gaseous fuel for a bounded MacCormack transport correction.
    const ph=this.advectedHeat, pf=this.advectedFuel;
    const {departure,departureX,departureY}=this;
    for(let y=0;y<NY;y++) for(let x=0,i=y*NX;x<NX;x++,i++) {
      const tx=clamp(x-dt*a.u[i],.001,NX-1.001), ty=clamp(y-dt*a.v[i],.001,NY-1.001);
      const ix=tx|0, iy=ty|0, n=ix+iy*NX;
      const fx=tx-ix, fy=ty-iy;
      departure[i]=n;departureX[i]=fx;departureY[i]=fy;
      ph[i]=interp(a.heat,n,fx,fy); pf[i]=interp(a.fuel,n,fx,fy);
    }
    const bpos=p.brush;
    const bx=bpos?bpos[0]*(NX-1):-999, by=bpos?bpos[1]*(NY-1):-999;
    if(wood)this.advanceWood(dt,ox,bx,by);
    const bed=this.wood, jetSpeed=jet?102:46, gasFeed=(jet?7.6:4.2)*p.fuel;
    const pilot=wood||p.coldStart?0:(jet?5.9:3.8)*p.fuel*unit(ox*2.4);
    for(let y=1;y<NY-1;y++) {
      const brushY=(y-by)**2, winds=16*y3[y], mid=y/NY, yLoss=.74+.42*mid*mid;
      for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
        const fi=departure[i], fx=departureX[i], fy=departureY[i];
        let u=interp(a.u,fi,fx,fy), v=interp(a.v,fi,fx,fy);
        const qx=clamp(x+dt*a.u[i],.001,NX-1.001), qy=clamp(y+dt*a.v[i],.001,NY-1.001);
        const qix=qx|0, qiy=qy|0, qi=qix+NX*qiy;
        let heat=corrected(a.heat,ph,i,fi,qi,qx-qix,qy-qiy);
        let fuel=corrected(a.fuel,pf,i,fi,qi,qx-qix,qy-qiy);
        let smoke=interp(a.smoke,fi,fx,fy), air=interp(a.air,fi,fx,fy);
        const noise=x1[x]*y1[y]+.49*x2[x]*y2[y]+.58*x3[x]*y3[y];
        const s=source[i]>0?source[i]*clamp(.94+.63*noise,.06,1.94):0;
        let brush=0;
        if(brushY<121){
          const dx=x-bx, distance=dx*dx+brushY;
          if(distance<121)brush=this.ignitionHeat[distance|0];
        }
        const feed=wood?bed.feed[x]*3.85:gasFeed;
        fuel=clamp(fuel+dt*s*feed,0,1.8);
        const surface=wood?bed.heat[x]*s*.47:0;
        heat+=dt*(s*pilot+surface*Math.max(0,1-heat*.72)+brush*6.8);
        air=unit(air+dt*(ox-air)*(.65+.22*Math.abs(noise)));
        const ignition=unit((heat-.064)*8);
        const reaction=Math.min(fuel,air*2.6,dt*fuel*Math.min(ox,air)*ignition*(5.75+1.4*Math.max(0,noise)));
        fuel-=reaction;
        air=Math.max(.01,air-reaction*.38);
        heat=Math.max(0,heat+reaction*2.94-dt*(yLoss*heat+.58*heat*heat));
        smoke=Math.max(0,smoke*(1-dt*(.35+.23*air*heat))+reaction*(.52+.46*(1-air))+s*dt*feed*.022);
        const gx=Math.abs(curl[i+1])-Math.abs(curl[i-1]), gy=Math.abs(curl[i+NX])-Math.abs(curl[i-NX]);
        const inv=1/(.04+Math.sqrt(gx*gx+gy*gy));
        const swirl=curl[i]*inv*(4.3*forces);
        const plume=unit(heat*2+smoke*2);
        u+=dt*(gy*swirl+(ambientWind-u)*.55+plume*forces*(winds+10*x1[x]*y1[y]+8*x2[x]*y2[y]));
        v+=dt*(-gx*swirl+(48*heat-3.3*smoke)-.37*v+plume*forces*(5*x3[x]-7*x2[x]*y1[y])+s*(jetSpeed-v)*(wood?1.15:2.8)*unit(.35*feed));
        b.u[i]=clamp(u,-94,94); b.v[i]=clamp(v,-65,138);
        const pattern=interp(a.filament,fi,fx,fy);
        const refresh=dt*(.13+s*3.1);
        b.filament[i]=pattern+(filamentSeed[i]-pattern)*refresh;
        b.fuel[i]=fuel; b.heat[i]=Math.min(2.4,heat); b.smoke[i]=Math.min(2,smoke); b.air[i]=air;
        burn[i]=reaction/dt;
      }
    }
    // Open sides and lid. Floor supplies ambient air to the burner.
    for(let y=0;y<NY;y++) {
      for(const x of [0,NX-1]) {
        const i=y*NX+x,n=i+(x===0?1:-1);
        b.u[i]=b.u[n]; b.v[i]=b.v[n]; b.air[i]=ox;
      }
    }
    for(let x=0;x<NX;x++) {
      const hi=(NY-1)*NX+x, k=hi-NX;
      b.air[x]=ox; b.heat[x]=b.heat[x+NX]*.36; b.fuel[x]=b.fuel[x+NX]*.74; b.smoke[x]=b.smoke[x+NX]*.34; b.v[x]=0; b.u[x]=b.u[x+NX]*.7;
      b.v[hi]=Math.max(0,b.v[k]); b.u[hi]=b.u[k]; b.air[hi]=b.air[k];
      b.filament[x]=b.filament[x+NX]; b.filament[hi]=b.filament[k];
      b.heat[hi]=b.heat[k]; b.smoke[hi]=b.smoke[k]; b.fuel[hi]=b.fuel[k];
    }
    for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
      divergence[i]=b.u[i]-b.u[i-1]+b.v[i]-b.v[i-NX];
    }
    for(let k=0;k<5;k++) {
      for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
        pressure[i]=.25*(pressure[i+1]+pressure[i-1]+pressure[i+NX]+pressure[i-NX]-divergence[i]);
      }
    }
    for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
      b.u[i]-=pressure[i+1]-pressure[i]; b.v[i]-=pressure[i+NX]-pressure[i];
    }
    this.a=b; this.b=a;this.revision++;
    let power=0, area=0, moment=0;
    for(let y=2;y<Math.ceil(NY*.16);y+=2) for(let x=4;x<NX-4;x+=2) {
      const i=x+y*NX, weight=source[i];
      const radiation=weight*unit((b.heat[i]-.14)*.65+burn[i]*.24);
      power+=radiation;moment+=radiation*x;area+=weight;
    }
    // Ground illumination averages a footprint whose response is slower than
    // gas emission. It keeps noisy single-cell jumps out of the scene lighting.
    const targetPower=area>0?unit(power/area):0;
    const targetX=power>1e-5?(moment/(NX*power)-.5):this.lightX;
    this.radiantPower+=(targetPower-this.radiantPower)*(1-Math.exp(-dt/.17));
    this.lightX+=(targetX-this.lightX)*(1-Math.exp(-dt/.24));
    this.sparkBudget+=dt*(wood?18*unit(.67*this.radiantPower+this.emberPower):3*this.radiantPower);
    while(this.sparkBudget>=1) {this.spawnSpark(); this.sparkBudget--;}
    for(let k=this.sparks.length-1;k>=0;k--) {
      const e=this.sparks[k];
      e.age+=dt;
      if(e.age>=e.life||e.y>flightBound||e.y<0||e.x<2||e.x>NX-3) {this.sparks.splice(k,1);continue;}
      const i=(e.x|0)+(e.y|0)*NX;
      e.x+=dt*(e.u+.7*b.u[i]+2*Math.sin(e.seed+t*4)); e.y+=dt*(e.v*.48+.64*b.v[i]);
    }
  }
  projection(options={}) {
    const w=this.canvas.width, h=this.canvas.height;
    const s=clamp(3.9/(options.distance||3.9),.62,1.8)*Math.min(h*.91,w*.99);
    return {s,ground:h*.84,cx:w*.5,w,h};
  }
  brushAt(pixelX,pixelY,options={}) {
    const {cx,ground,s}=this.projection(options);
    return [unit((pixelX-cx)/s+.5),unit((ground-pixelY)/(s*1.12)),.5];
  }
  prepareSlice(quality) {
    const scale=quality==='ultra'?2:quality==='high'?1.5:quality==='low'?.90:1.20;
    const width=Math.round(NX*scale), height=Math.round(NY*scale);
    if(this.slice.width===width && this.sampleX) return;
    this.slice.width=width; this.slice.height=height;
    this.image=this.sliceContext.createImageData(width,height);
    this.imageRevision=-1;
    this.sampleX=new Int16Array(width); this.fractX=new Float32Array(width);
    for(let x=0;x<width;x++) {
      const sx=clamp((x+.5)/scale-.5,0,NX-1.001);
      this.sampleX[x]=sx|0; this.fractX[x]=sx-(sx|0);
    }
    this.rowY=new Int32Array(height);this.fractY=new Float32Array(height);
    this.fadeY=new Float32Array(height);
    this.fadeX=Float32Array.from(this.sampleX,(i,j)=>{
      const x=i+this.fractX[j];
      return ramp(0,8,Math.min(x,NX-1-x));
    });
    for(let y=0;y<height;y++){
      const sy=clamp((y+.5)/scale-.5,0,NY-1.001);
      this.rowY[y]=(sy|0)*NX;this.fractY[y]=sy-(sy|0);
      this.fadeY[y]=ramp(0,14,NY-1-sy);
    }
  }
  updateRadiance() {
    const {a,burn,luminance,sheets,curl}=this;
    const heat=a.heat, fuel=a.fuel;
    // Leave an almost transparent hot wake. Peak emission sits on the steep,
    // reactant-rich boundaries where oxygen mixes with the pyrolysis gases.
    for(let y=1,i=NX+1;y<NY-1;y++,i+=2) for(let x=1;x<NX-1;x++,i++) {
      const h=heat[i];
      if(h<.055){luminance[i]=0;sheets[i]=0;continue;}
      const tx=heat[i+1]-heat[i-1], ty=heat[i+NX]-heat[i-NX];
      const fx=fuel[i+1]-fuel[i-1], fy=fuel[i+NX]-fuel[i-NX];
      const gradient=Math.sqrt(tx*tx+ty*ty);
      const mixGradient=Math.sqrt(fx*fx+fy*fy);
      const front=unit(1.16*gradient+.72*mixGradient);
      const hot=unit((h-.055)/2.345);
      const reaction=unit(burn[i]*.38);
      const swirl=unit(Math.abs(curl[i])*.26);
      const scatter=1-.24*unit(fuel[i]*.71)*unit(a.smoke[i]*1.72);
      const filament=1+.76*a.filament[i];
      // A broad amber volume surrounds narrower active flame surfaces.
      luminance[i]=Math.sqrt(hot)*(.54+.37*hot)*(.46+.61*front)*scatter;
      sheets[i]=hot*(.10+.72*front*front+.46*reaction+.39*swirl*swirl)*filament*scatter;
    }
  }
  paintFire(options) {
    this.prepareSlice(options.quality);
    const mode=options.mode||'beauty', exposure=clamp(Number(options.exposure??1),.12,4);
    if(this.imageRevision===this.revision && this.imageMode===mode && this.imageExposure===exposure)return;
    const {a,image,emission,luminance,sheets,opacity,sampleX,fractX,rowY,fractY,fadeX,fadeY}=this;
    const pixels=image.data, w=this.slice.width, h=this.slice.height;
    const {heat,smoke,filament}=a;
    const beauty=mode==='beauty', smokeView=mode==='smoke', tempView=mode==='temperature';
    const jet=this.params.preset==='jet', velocityView=mode==='velocity';
    const fine=options.quality==='high'||options.quality==='ultra';
    if(beauty)this.updateRadiance();
    for(let y=0;y<h;y++) {
      const fy=fractY[y], row=rowY[y], height=y/h, fade=fadeY[y];
      const lowerLight=unit((1-height*1.64)*this.radiantPower);
      const blueBase=jet?unit((.235-height)*8.6):0;
      for(let x=0,j=(h-1-y)*w*4;x<w;x++,j+=4) {
        const fx=fractX[x], i=row+sampleX[x];
        if(velocityView) {
          const u=interp(a.u,i,fx,fy),v=interp(a.v,i,fx,fy),q=unit(Math.sqrt(u*u+v*v)/74);
          pixels[j]=8+200*q*q; pixels[j+1]=35+195*q;
          pixels[j+2]=68+130*(1-q); pixels[j+3]=220*unit(q*2.4);
          continue;
        }
        const soot=interp(smoke,i,fx,fy);
        if(tempView) {
          const t=interp(heat,i,fx,fy), k=unit(t*.7);
          pixels[j]=255*unit(1.35*k); pixels[j+1]=255*unit(2*k-.66);
          pixels[j+2]=255*unit(.75*Math.exp(-8*(k-.24)**2)+unit(3*k-2));
          pixels[j+3]=255*unit(t*2+soot);continue;
        }
        const lum=beauty?interp(luminance,i,fx,fy):0;
        const sheet=lum>.014?(fine?cubicFront(sheets,i,fx,fy):interp(sheets,i,fx,fy)):0;
        if(lum<.001 && soot<.003){pixels[j+3]=0;continue;}
        const pattern=interp(filament,i,fx,fy);
        const plume=soot*(.83+.28*pattern);
        const flameAlpha=opacityAt(opacity,lum*1120+sheet*590);
        const smokeAlpha=opacityAt(opacity,plume*(smokeView?940:382))*(1-.68*flameAlpha);
        const alpha=flameAlpha+smokeAlpha*(1-flameAlpha);
        if(alpha<.003){pixels[j+3]=0;continue;}
        const f=flameAlpha/alpha, sm=1-f;
        const litEdge=unit(.22+.28*pattern+.53*lum);
        const ash=smokeView?101+22*pattern:28+24*litEdge+14*pattern;
        const hotFog=lowerLight*(25+37*litEdge+10*pattern);
        let r=0,g=0,b=0;
        if(f>0) {
          const temp=interp(heat,i,fx,fy);
                    const temperature=clamp(31*(temp-.045)+34*sheet+16*Math.sqrt(unit(temp)),0,126.999), tBin=temperature|0, warmth=temperature-tBin;
          const energy=Math.min(126.999,(.92*lum+.84*sheet)*exposure*64), bin=energy|0, blend=energy-bin;
          const c=3*(128*tBin+bin), h=c+384;
          const red=emission[c]+blend*(emission[c+3]-emission[c]);
          const green=emission[c+1]+blend*(emission[c+4]-emission[c+1]);
          const blue=emission[c+2]+blend*(emission[c+5]-emission[c+2]);
          r=red+warmth*(emission[h]+blend*(emission[h+3]-emission[h])-red);
          g=green+warmth*(emission[h+1]+blend*(emission[h+4]-emission[h+1])-green);
          b=blue+warmth*(emission[h+2]+blend*(emission[h+5]-emission[h+2])-blue);
        }
        const blue=blueBase*unit(1-1.25*soot)*unit(1.5*lum*exposure)*f;
        pixels[j]=r*(f-blue)+(ash+hotFog)*sm+85*blue;
        pixels[j+1]=g*(f-blue)+(ash+hotFog*.53)*sm+151*blue;
        pixels[j+2]=b*(f-blue)+(ash*1.075+hotFog*.28)*sm+255*blue;
        pixels[j+3]=255*alpha*fade*fadeX[x];
      }
    }
    this.sliceContext.putImageData(image,0,0);
    this.imageRevision=this.revision;this.imageMode=mode;this.imageExposure=exposure;
  }
  buildScene(projection) {
    const {s,ground,cx,w,h}=projection, wood=isWood(this.params.preset);
    const key=[w,h,s,this.params.preset].join('/');
    if(this.sceneCache?.key===key)return this.sceneCache;
    const back=document.createElement('canvas');back.width=w;back.height=h;
    const c=back.getContext('2d',{alpha:false});
    c.fillStyle='#0b0d11'; c.fillRect(0,0,w,h);
    c.translate(cx,ground); c.scale(s,s);
    c.save();c.scale(1,.43);
    c.fillStyle=grad(c,0,.03,1.22,[[0,'#38312d'],[.32,'#242225'],[1,'#0b0d11']]);
    c.fillRect(-1.18,-1.18,2.36,2.36);c.restore();
    const rand=randomSequence(98152);
    for(let i=0;i<1400;i++) {
      const x=(rand()-.5)*2.8, depth=rand()**2;
      const y=.08-(1-depth)**2*.20+depth**4*.64;
      const ox=x*(.6+depth), r=Math.max(.65/s,.0006+.0032*depth*depth);
      const light=1/(1+6*ox*ox+22*y*y);
      c.fillStyle='rgba('+Math.round(54+38*light)+','+Math.round(52+35*light)+',56,'+(.04+rand()*.16)+')';
      c.beginPath();c.ellipse(ox,y,r*(1+1.4*rand()),Math.max(.46/s,r*.47),0,0,Math.PI*2);c.fill();
    }
    // The ground light layer is static. Its location and energy track simulated combustion.
    const light=document.createElement('canvas');light.width=256;light.height=128;
    const l=light.getContext('2d');
    l.save();l.translate(128,64);l.scale(1,.50);
    l.fillStyle=grad(l,0,0,128,[[0,'rgba(231,116,56,.71)'],[.22,'rgba(187,82,39,.48)'],[.56,'rgba(103,47,27,.24)'],[1,'rgba(94,24,10,0)']]);
    l.fillRect(-128,-128,256,256);l.restore();
    const layers={key,back,light};
    const backLogs=[
      [-.247,-.003,.223,-.054,.033,17],
      [-.182,-.088,.238,-.012,.040,39],
      [-.255,-.045,.129,-.057,.022,47]
    ];
    const frontLogs=[
      [-.272,-.016,.162,.032,.029,7],
      [.029,.027,.260,-.050,.026,54]
    ];
    // Both fixture passes share a tight bounding box so drawing three layers
    // adds little canvas bandwidth even when the viewport is large.
    for(const name of ['rear','front','rearLight','frontLight']) {
      const canvas=document.createElement('canvas');
      canvas.width=Math.ceil(.75*s);canvas.height=Math.ceil(.38*s);
      const g=canvas.getContext('2d');g.translate(.375*s,.232*s);g.scale(s,s);
      if(wood) {
        if(name==='rear') {
          g.save();g.scale(1,.22);
          g.fillStyle=grad(g,0,.029,.39,[[0,'rgba(3,3,5,.88)'],[.57,'rgba(3,3,5,.66)'],[1,'rgba(3,3,5,0)']]);
          g.fillRect(-.38,-.35,.76,.82);g.restore();
          this.drawAsh(g,false);
          for(const log of backLogs)this.drawLog(g,...log);
        } else if(name==='front') {
          for(const log of frontLogs)this.drawLog(g,...log);
        } else {
          if(name==='rearLight')this.drawAsh(g,true);
          const logs=name==='rearLight'?backLogs:frontLogs;
          for(const log of logs){
            this.drawLog(g,...log,true);
            this.coalCracks(g,...log);
          }
        }
      } else if(name==='rear')this.drawBurner(g,this.params.preset==='jet');
      layers[name]=canvas;
    }
    this.sceneCache=layers;
    return layers;
  }
  drawAsh(c,glowLayer) {
    const rand=randomSequence(68212);
    for(let i=0;i<430;i++) {
      const a=rand()*Math.PI*2, radius=Math.sqrt(rand());
      const x=.292*radius*Math.cos(a), y=.062*radius*Math.sin(a)-.014;
      const scale=.0012+.014*rand()**2, shade=28+rand()*23, coal=rand();
      if(glowLayer && (coal>.38||radius>.92))continue;
      c.fillStyle=glowLayer?'rgba(255,'+Math.round(54+108*coal)+',19,'+(.18+coal)+')':
        'rgb('+Math.round(shade+7)+','+Math.round(shade+3)+','+Math.round(shade)+')';
      c.beginPath();c.moveTo(x-scale,y);c.lineTo(x-.37*scale,y-.45*scale);c.lineTo(x+.47*scale,y-.64*scale);
      c.lineTo(x+scale,y+.15*scale);c.lineTo(x-.45*scale,y+.52*scale);c.fill();
    }
  }
  coalCracks(c,x0,y0,x1,y1,r,seed) {
    const random=randomSequence(seed*5831), length=Math.hypot(x1-x0,y1-y0);
    c.save();c.translate(x0,y0);c.rotate(Math.atan2(y1-y0,x1-x0));
    for(let i=0;i<83;i++) {
      const x=.003+(length-.006)*random(), worldX=x0+x*(x1-x0)/length;
      const span=Math.min(length-x,.002+.041*random()**2), y=(random()*1.61-.83)*r;
      const central=unit(1-Math.abs(worldX)/.275);
      if(random()>.86*central)continue;
      const red=54+Math.round(114*random()), grain=.14+.58*central*random();
      c.strokeStyle='rgba(255,'+red+',16,'+grain+')';
      c.lineWidth=.00083+random()*.00089;
      c.beginPath();c.moveTo(x,y);c.lineTo(x+span*.42,y+.003*random());
      c.lineTo(x+.66*span,y-.004*random());c.lineTo(x+span,y+.002*random());c.stroke();
    }
    c.restore();
  }
  draw(options={}) {
    this.paintFire(options);
    const projection=this.projection(options),{s,ground,cx}=projection;
    const {back,light,rear,front,rearLight,frontLight}=this.buildScene(projection),c=this.ctx;
    const beauty=!options.mode||options.mode==='beauty';
    const fx=Math.round(cx-s*.375), fy=Math.round(ground-s*.232);
    c.drawImage(back,0,0);
    if(beauty && this.radiantPower>.002) {
      c.globalAlpha=.69*this.radiantPower;
      c.drawImage(light,cx-s*(.54-this.lightX),ground-s*.24,1.08*s,.48*s);
      c.globalAlpha=1;
    }
    const coals=beauty?unit(1.27*this.emberPower+.30*this.radiantPower):0;
    c.drawImage(rear,fx,fy);
    if(coals>.004){
      c.globalAlpha=coals;
      c.drawImage(rearLight,fx,fy);
      c.globalAlpha=1;
    }
    c.imageSmoothingEnabled=true;
    c.imageSmoothingQuality=options.quality==='low'?'low':'medium';
    c.drawImage(this.slice,cx-s*.5,ground-s*1.12,s,s*1.12);
    c.drawImage(front,fx,fy);
    if(coals>.004) {
      c.globalAlpha=.76*coals;
      c.drawImage(frontLight,fx,fy);
      c.globalAlpha=1;
    }
    if(beauty) {
      c.lineCap='round';
      for(const e of this.sparks) {
        const alpha=.78*ramp(0,.10,e.age)*ramp(0,.76,e.life-e.age);
        if(alpha<.01)continue;
        c.strokeStyle='rgba(255,'+Math.round(132+94*alpha)+',66,'+alpha+')';
        c.lineWidth=Math.max(.62,.0014*s*e.size);
        const x=cx+(e.x/NX-.5)*s, y=ground-e.y/NY*1.12*s;
        const ex=clamp(e.x,0,NX-1.001), ey=clamp(e.y,0,NY-1.001);
        const ix=ex|0, iy=ey|0, i=ix+iy*NX;
        const uv=interp(this.a.u,i,ex-ix,ey-iy)*.00012;
        const vv=interp(this.a.v,i,ex-ix,ey-iy)*.00010;
        c.beginPath();c.moveTo(x,y);c.lineTo(x-uv*s,y+(.0012+e.size*.001+Math.max(0,vv))*s);c.stroke();
      }
    }
  }
  drawBurner(c,jet) {
    const r=jet?.065:.31, side=jet?.071:.028, top=jet?-.073:-.018;
    c.fillStyle='#16191b'; c.beginPath(); c.ellipse(0,top+side,r,r*.155,0,0,Math.PI*2); c.fill();
    const g=c.createLinearGradient(-r,0,r,0);
    g.addColorStop(0,'#292b2b'); g.addColorStop(.26,'#383733'); g.addColorStop(.69,'#222323'); g.addColorStop(1,'#131415');
    c.fillStyle=g; c.fillRect(-r,top,2*r,side);
    c.fillStyle='#2d201a'; c.beginPath(); c.ellipse(0,top,r,r*.155,0,0,Math.PI*2); c.fill();
    c.strokeStyle='rgba(161,125,103,.39)'; c.lineWidth=.0026; c.stroke();
  }
  drawLog(c,x0,y0,x1,y1,r,seed,lit=false) {
    const length=Math.hypot(x1-x0,y1-y0), rand=randomSequence(seed);
    c.save(); c.translate(x0,y0); c.rotate(Math.atan2(y1-y0,x1-x0));
    // Irregular break points and bark thickness, shared by the warm-light mask.
    const outer=Array.from({length:15},(_,i)=>{
      const t=unit((i+(i===0||i===14?0:.75*(rand()-.5)))/14);
      const taper=.83+.14*Math.sin(t*2.7+seed);
      return [length*t,-r*(taper+.22*rand()),r*(taper+.20*rand())];
    });
    c.beginPath();c.moveTo(outer[0][0],outer[0][1]);
    for(const [x,up] of outer)c.lineTo(x,up);
    for(let i=outer.length-1;i>=0;i--)c.lineTo(outer[i][0],outer[i][2]);
    c.closePath();
    if(lit) {
      // Warm spill fades over the outer bark and toward either end.
      c.clip();
      c.save();c.translate(length*.56,-r*.63);c.scale(1,.36);
      c.fillStyle=grad(c,0,0,length*.68,[
        [0,'rgba(224,111,50,.38)'],[.42,'rgba(163,68,32,.25)'],
        [.80,'rgba(96,37,19,.16)'],[1,'rgba(55,19,10,0)']
      ]);
      c.fillRect(-length*.75,-length*.75,length*1.5,length*1.5);
      c.restore();c.restore();return;
    }
    const grain=c.createLinearGradient(0,-r,0,r);
    grain.addColorStop(0,'#653d27'); grain.addColorStop(.18,'#473026');
    grain.addColorStop(.48,'#2c211c'); grain.addColorStop(1,'#0f1012');
    c.fillStyle=grain;c.fill();
    c.save();c.clip();
    for(let i=0;i<110;i++) {
      const x=.003+rand()*(length-.013), y=(rand()*1.63-.81)*r;
      const run=.003+rand()*.040, thick=.0012+rand()*.0046;
      const facing=unit(.50-y/(2*r)+.24*(rand()-.5));
      const shade=Math.round(18+facing*27+rand()*22);
      c.fillStyle='rgb('+shade+','+Math.round(shade*.80)+','+Math.round(shade*.72)+')';
      c.beginPath();c.moveTo(x,y);c.lineTo(x+.32*run,y-thick);c.lineTo(x+run,y-.72*thick);
      c.lineTo(x+.90*run,y+thick);c.lineTo(x+.14*run,y+.44*thick);c.fill();
    }
    for(let i=0;i<49;i++) {
      const x=rand()*length, y=(rand()*1.7-.9)*r, run=Math.min(length-x,.008+rand()*.061);
      c.strokeStyle=rand()<.57?'rgba(8,8,9,.69)':'rgba(162,115,87,.34)';
      c.lineWidth=.0013+rand()*.002; c.beginPath();c.moveTo(x,y);
      c.lineTo(x+run*.49,y+rand()*.006-.003);c.lineTo(x+run,y+rand()*.007-.0035);c.stroke();
    }
    c.restore();
    // An uneven cut end with broad bands and radial splits. No equispaced rings.
    const cut=c.createLinearGradient(-r*.44,-r,r*.44,r);
    cut.addColorStop(0,'#785139');cut.addColorStop(.45,'#422918');cut.addColorStop(1,'#151416');
    c.fillStyle=cut;c.beginPath();c.ellipse(0,0,r*.39,r*.94,-.06,0,Math.PI*2);c.fill();
    for(const band of [.36,.65,.90]){
      c.strokeStyle='rgba(157,94,51,.42)';c.lineWidth=.0019;
      c.beginPath();
      for(let j=0;j<=13;j++){
        const angle=j*Math.PI*2/13, edge=band*(.87+.20*rand());
        const x=r*.35*Math.cos(angle)*edge, y=r*.89*Math.sin(angle)*edge;
        if(j)c.lineTo(x,y);else c.moveTo(x,y);
      }
      c.closePath();c.stroke();
    }
    for(let i=0;i<6;i++){
      const angle=rand()*2*Math.PI, cx=r*.34*Math.cos(angle), cy=r*.87*Math.sin(angle);
      c.strokeStyle='rgba(13,9,8,.66)';c.lineWidth=.0011+rand()*.0015;
      c.beginPath();c.moveTo(cx,cy);c.lineTo(cx*.34+.003*(rand()-.5),cy*.46);c.stroke();
    }
    c.restore();
  }
  destroy(){this.sparks.length=0;this.sceneCache=null;}
}

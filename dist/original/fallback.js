// Reduced 2D path. The velocity projection, transported species, heat release,
// and oxygen consumption all run in the CPU solver on the same Eulerian grid.
const NX = 104, NY = 124, N = NX * NY;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const unit = x => clamp(x, 0, 1);
const defaults = {preset: 'wood', fuel: .66, oxygen: .78, turbulence: .58, wind: 0};
const burners = {
  wood: [[-.50,.07,.65],[-.35,.10,1],[-.18,.14,1.12],[.015,.12,1.1],[.21,.12,1.15],[.43,.085,.86]],
  pool: [[-.53,.11,.8],[-.32,.14,1.1],[-.1,.16,.97],[.12,.14,1.16],[.34,.14,1.1],[.53,.09,.84]],
  jet: [[-.045,.06,1.2],[.035,.055,1.1]],
  crosswind: [[-.48,.09,.91],[-.29,.13,1.1],[-.12,.11,.85],[.07,.10,1.12],[.24,.08,.8]]
};
const palette = [
  [0,0,0],[42,9,6],[100,24,8],[165,42,7],[218,64,9],[243,90,10],
  [253,125,18],[255,158,34],[255,185,62],[255,210,103],[255,229,151],[255,244,208]
];
function colorTable() {
  const a = new Uint8Array(256*3);
  for(let i=0;i<256;i++) {
    const t = (i/255)*(palette.length-1), j = Math.min(palette.length-2, Math.floor(t));
    for(let c=0;c<3;c++) a[i*3+c] = palette[j][c]+(palette[j+1][c]-palette[j][c])*(t-j);
  }
  return a;
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
    this.pressure=new Float32Array(N); this.divergence=new Float32Array(N);
    this.source=new Float32Array(N); this.burn=new Float32Array(N);
    this.xWave=[new Float32Array(NX),new Float32Array(NX),new Float32Array(NX)];
    this.yWave=[new Float32Array(NY),new Float32Array(NY),new Float32Array(NY)];
    this.palette=colorTable();
    this.opacity=Float32Array.from({length:2049},(_,i)=>1-Math.exp(-i/256));
    this.colors=Uint16Array.from({length:2049},(_,i)=> {
      const energy=i/512;
      return 3*(255*Math.pow(unit(energy/(.58+.84*energy)),.80)|0);
    });
    this.phase=Float32Array.from({length:1024},(_,i)=>Math.sin(i*Math.PI/512));
    this.luminance=new Float32Array(N); this.sceneCache=new Map();this.cacheScene='';
    this.sparks=[];
    this.reset();
  }
  reset(params={}) {
    this.params={...defaults,...params};
    const kind=this.params.preset;
    const nozzle=kind==='jet';
    const burning=clamp(this.params.fuel,0,1)*clamp(this.params.oxygen*2,0,1);
    const sources=burners[kind]||burners.wood;
    this.random=randomSequence(938102);
    this.radiantPower=burning; this.sparks.length=0; this.time=0; this.sparkBudget=0;
    this.burn.fill(0); this.pressure.fill(0); this.curl.fill(0);
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
      const ripple=.83+.13*Math.sin(29*xx+17*h)+.13*Math.sin(43*h-27*xx);
      const plume=Math.exp(-1*((xx-drift-.075*Math.sin(15*h))/(.26+.20*h))**2);
      this.source[i]=Math.min(1.8,src)*(nozzle?1:.42+.58*(.5+.5*Math.cos(xx*39+.7*Math.sin(xx*20))));
      this.a.heat[i]=Math.min(1.65,flame*1.45*ripple)*clamp(1.61*burning,0,1.55);
      this.a.fuel[i]=Math.min(.7,flame*.48+src*.22)*this.params.fuel;
      this.a.smoke[i]=burning*(.12+.56*h)*plume*Math.sin(Math.PI*h)**.5;
      // Material coordinate follows warped source streamlines through the initial plume.
      const shifted=xx-.09*Math.sin(16*h+xx*4.8)-.04*Math.sin(31*h-xx*6.3);
      this.a.filament[i]=.52*shifted+.15*Math.sin(7*shifted-8*h)+.08*Math.cos(9*shifted+14*h);
      this.a.air[i]=this.params.oxygen*(1-Math.min(.53,flame*.23));
      this.a.u[i]=7*Math.sin(h*11)*plume+this.params.wind*13;
      this.a.v[i]=(nozzle?55:38)*plume*Math.max(.05,(1-h*.8));
      this.burn[i]=Math.min(1,flame)*.7;
    }
    for(let i=0;i<Math.round(burning*34);i++) this.spawnSpark(this.random()*.72);
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
    const {a,b,curl,divergence,pressure,burn,source}=this;
    const p=this.params, t=this.time, jet=p.preset==='jet'||Boolean(this.reducedNozzle?.()), ox=unit(p.oxygen);
    const forces=.25+1.35*p.turbulence;
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
    for(let y=0;y<NY;y++) for(let x=0,i=y*NX;x<NX;x++,i++) {
      const tx=clamp(x-dt*a.u[i],.001,NX-1.001), ty=clamp(y-dt*a.v[i],.001,NY-1.001);
      const ix=tx|0, iy=ty|0, n=ix+iy*NX;
      ph[i]=interp(a.heat,n,tx-ix,ty-iy); pf[i]=interp(a.fuel,n,tx-ix,ty-iy);
    }
    const gains=this.burnerGains?.(p,t);
    const inletTime=a.inletTime, nextTime=b.inletTime;
    const bpos=p.brush;
    const bx=bpos?bpos[0]*(NX-1):-999, by=bpos?bpos[1]*(NY-1):-999;
    for(let y=1;y<NY-1;y++) {
      for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
        const tx=clamp(x-dt*a.u[i],.001,NX-1.001), ty=clamp(y-dt*a.v[i],.001,NY-1.001);
        const ix=tx|0, iy=ty|0, fi=ix+iy*NX, fx=tx-ix, fy=ty-iy;
        let u=interp(a.u,fi,fx,fy), v=interp(a.v,fi,fx,fy);
        const qx=clamp(x+dt*a.u[i],.001,NX-1.001), qy=clamp(y+dt*a.v[i],.001,NY-1.001);
        const qix=qx|0, qiy=qy|0, qi=qix+NX*qiy;
        let heat=corrected(a.heat,ph,i,fi,qi,qx-qix,qy-qiy);
        let fuel=corrected(a.fuel,pf,i,fi,qi,qx-qix,qy-qiy);
        let smoke=interp(a.smoke,fi,fx,fy), air=interp(a.air,fi,fx,fy);
        const noise=x1[x]*y1[y]+.49*x2[x]*y2[y]+.58*x3[x]*y3[y];
        let s=source[i]*clamp(.94+.63*noise,.06,1.94);
        if(gains) s*=gains[x];
        const dd=(x-bx)**2+(y-by)**2;
        if(dd<100) s+=2.2*(1-dd/100)**2;
        const jetSpeed=jet?102:46;
        fuel=clamp(fuel+dt*s*((jet?7.6:4.2)*p.fuel),0,1.8);
        heat+=dt*s*((jet?5.9:3.8)*p.fuel)*unit(ox*2.4);
        air=unit(air+dt*(ox-air)*(.65+.22*Math.abs(noise)));
        const ignition=unit((heat-.064)*8);
        const reaction=Math.min(fuel,air*2.6,dt*fuel*air*ignition*(5.4+1.4*Math.max(0,noise)));
        fuel-=reaction;
        air=Math.max(.01,air-reaction*.38);
        heat=Math.max(0,heat+reaction*2.94-dt*((.74+.42*(y/NY)**2)*heat+.58*heat*heat));
        smoke=Math.max(0,smoke*(1-dt*(.35+.23*air*heat))+reaction*(.56+.45*(1-air))+s*dt*.14*p.fuel);
        const gx=Math.abs(curl[i+1])-Math.abs(curl[i-1]), gy=Math.abs(curl[i+NX])-Math.abs(curl[i-NX]);
        const inv=1/(.04+Math.sqrt(gx*gx+gy*gy));
        const swirl=curl[i]*inv*(4.3*forces);
        const plume=unit(heat*2+smoke*2);
        u+=dt*(gy*swirl+(p.wind*30-u)*.55+plume*forces*(16*y3[y]+10*x1[x]*y1[y]+8*x2[x]*y2[y]));
        v+=dt*(-gx*swirl+(48*heat-3.3*smoke)-.37*v+plume*forces*(5*x3[x]-7*x2[x]*y1[y])+s*(jetSpeed-v)*2.8*p.fuel);
        b.u[i]=clamp(u,-94,94); b.v[i]=clamp(v,-65,138);
        const material=interp(a.filament,fi,fx,fy);
        const incoming=.97*(x/NX-.5)+.12*x1[x]+.07*x2[x];
        if(inletTime) {
          const stamp=interp(inletTime,fi,fx,fy);
          nextTime[i]=stamp+(t-stamp)*Math.min(1,dt*s*7.0);
        }
        b.filament[i]=material+(incoming-material)*Math.min(1,dt*s*7.0);
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
      divergence[i]=.5*(b.u[i+1]-b.u[i-1]+b.v[i+NX]-b.v[i-NX]);
    }
    this.applyThermalExpansion?.(divergence,b.heat,burn,dt,b);
    for(let k=0;k<5;k++) {
      for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
        pressure[i]=.25*(pressure[i+1]+pressure[i-1]+pressure[i+NX]+pressure[i-NX]-divergence[i]);
      }
    }
    for(let y=1;y<NY-1;y++) for(let x=1,i=y*NX+1;x<NX-1;x++,i++) {
      b.u[i]-=.5*(pressure[i+1]-pressure[i-1]); b.v[i]-=.5*(pressure[i+NX]-pressure[i-NX]);
    }
    this.a=b; this.b=a;
    let power=0, area=0;
    for(let y=2;y<Math.ceil(NY*.16);y+=2) for(let x=4;x<NX-4;x+=2) {
      const i=x+y*NX, weight=source[i];
      power+=weight*unit((b.heat[i]-.14)*.65+burn[i]*.24);area+=weight;
    }
    this.radiantPower=area>0?unit(power/area):0;
    this.sparkBudget+=dt*(p.preset==='wood'||p.preset==='crosswind'?16:3)*this.radiantPower;
    while(this.sparkBudget>=1) {this.spawnSpark(); this.sparkBudget--;}
    for(let k=this.sparks.length-1;k>=0;k--) {
      const e=this.sparks[k];
      e.age+=dt;
      if(e.age>=e.life||e.y>NY-3||e.x<2||e.x>NX-3) {this.sparks.splice(k,1);continue;}
      const i=(e.x|0)+(e.y|0)*NX;
      e.x+=dt*(e.u+.7*b.u[i]+2*Math.sin(e.seed+t*4)); e.y+=dt*(e.v*.48+.64*b.v[i]);
    }
  }
  projection(options={},output=this.canvas) {
    const w=output.width, h=output.height;
    const s=clamp(3.9/(options.distance||3.9),.62,1.8)*Math.min(h*.91,w*.99);
    return {s,ground:h*.84,cx:w*.5,w,h};
  }
  brushAt(pixelX,pixelY,options={}) {
    const {cx,ground,s,gasHeight=s*1.12}=this.projection(options);
    return [unit((pixelX-cx)/s+.5),unit((ground-pixelY)/gasHeight),.5];
  }
  prepareSlice(quality) {
    const scale=quality==='ultra'?2.5:quality==='high'?2:quality==='low'?1:1.5;
    const width=Math.round(NX*scale), height=Math.round(NY*scale);
    if(this.slice.width===width && this.sampleX) return;
    this.slice.width=width; this.slice.height=height;
    this.image=this.sliceContext.createImageData(width,height);
    this.sampleX=new Int16Array(width); this.fractX=new Float32Array(width);
    for(let x=0;x<width;x++) {
      const sx=clamp((x+.5)/scale-.5,0,NX-1.001);
      this.sampleX[x]=sx|0; this.fractX[x]=sx-(sx|0);
    }
    this.sliceScale=scale;
  }
  paintFire(options) {
    this.prepareSlice(options.quality);
    const {a,burn,image,palette,phase,luminance,opacity,colors}=this;
    const pixels=image.data, mode=options.mode||'beauty';
    const exposure=clamp(Number(options.exposure??1),.12,4);
    const w=this.slice.width, h=this.slice.height, scale=this.sliceScale;
    const heat=a.heat, smoke=a.smoke, material=a.filament;
    const filtered=mode==='beauty' && options.stableSampling!==false && this.filterFold;
    const shared=mode==='beauty' && options.sharedLighting!==false && this.optics;
    if(mode==='beauty') {
      for(let y=0,i=0;y<NY;y++) for(let x=0;x<NX;x++,i++) {
        const t=heat[i];
        if(t<.09){luminance[i]=0;continue;}
        const cx=clamp(x,1,NX-2), cy=clamp(y,1,NY-2), k=cy*NX+cx;
        // Steep temperature gradients delineate the flame front. Local mixture
        // and heat release shape its emission on the oxidizer-facing surface.
        const front=Math.abs(heat[k+1]-heat[k-1])+Math.abs(heat[k+NX]-heat[k-NX]);
        const mixture=a.fuel[i]*a.air[i], rate=unit(burn[i]*.8);
        const sheet=unit(front*1.25+mixture*1.8);
        luminance[i]=(t-.083)*(.54+.14*rate+.55*sheet);
      }
    }
    if(shared)shared.update(a,burn,luminance,this.params.preset,this.time);
    const opticalFront=shared?shared.emissionFront:luminance;
    for(let y=0;y<h;y++) {
      const sy=clamp((y+.5)/scale-.5,0,NY-1.001), iy=sy|0, fy=sy-iy, row=iy*NX;
      if(shared)shared.prepareRow(iy,fy,this.sampleX,this.fractX);
      for(let x=0,j=(h-1-y)*w*4;x<w;x++,j+=4) {
        const ix=this.sampleX[x], fx=this.fractX[x], i=row+ix;
        const soot=interp(smoke,i,fx,fy);
        if(mode==='velocity') {
          const u=interp(a.u,i,fx,fy),v=interp(a.v,i,fx,fy),q=unit(Math.sqrt(u*u+v*v)/74);
          pixels[j]=8+200*q*q; pixels[j+1]=35+195*q;
          pixels[j+2]=68+130*(1-q); pixels[j+3]=220*unit(q*2.4);
          continue;
        }
        if(mode==='temperature') {
          const t=interp(heat,i,fx,fy), k=unit(t*.7);
          pixels[j]=255*unit(1.35*k); pixels[j+1]=255*unit(2*k-.66);
          pixels[j+2]=255*unit(.75*Math.exp(-8*(k-.24)**2)+unit(3*k-2));
          pixels[j+3]=255*unit(t*2+soot);continue;
        }
        let luminous=mode==='smoke'?0:interp(opticalFront,i,fx,fy);
        let fold=1;
        if(luminous>.002 && filtered){
          fold=this.filterFold(material,i,fx,fy,sy,scale);
        }else if(luminous>.002) {
          // An advected material coordinate puts sub-cell lamellae on folded
          // sheets. Coordinates stay attached to the moving flow through time.
          const m=interp(material,i,fx,fy);
          const p=m*6973+sy*2.4+phase[((m*3680+sy*8)|0)&1023]*112;
          const broad=phase[(p|0)&1023], fine=phase[((p*2.71+m*3120)|0)&1023];
          const ridge=unit((broad+.28*fine+.44)*1.55);
          fold=.78+.31*ridge+.10*fine*ridge;
        }
        luminous*=fold;
        const flameAlpha=opacity[Math.min(2048,luminous*1485)|0];
        const smokeAlpha=opacity[Math.min(2048,soot*(mode==='smoke'?692:435))|0]*(1-flameAlpha);
        const alpha=smokeAlpha+flameAlpha;
        if(alpha<.003){pixels[j+3]=0;continue;}
        if(shared){
          shared.shade(pixels,j,x,ix,fx,soot,flameAlpha/alpha,fold,exposure);
          pixels[j+3]=255*alpha*unit((NY-1-sy)/15)*unit(Math.min(ix,NX-1-ix)/8);
          continue;
        }
        const c=colors[Math.min(2048,luminous*exposure*512)|0];
        const sColor=(mode==='smoke'?92:48)+25*unit(luminous*1.6)+17*sy/NY;
        const f=flameAlpha/alpha, sm=1-f;
        pixels[j]=palette[c]*f+sColor*sm;
        pixels[j+1]=palette[c+1]*f+(sColor-1)*sm;
        pixels[j+2]=palette[c+2]*f+(sColor+1)*sm;
        pixels[j+3]=255*alpha*unit((NY-1-sy)/15)*unit(Math.min(ix,NX-1-ix)/8);
      }
    }
    this.sliceContext.putImageData(image,0,0);
  }
  buildScene(options,projection) {
    const {s,ground,cx,w,h}=projection;
    const beauty=options.mode==='beauty'||!options.mode;
    const shared=beauty && this.optics && options.sharedLighting!==false;
    const level=shared?2:Math.round(this.radiantPower*5), glow=level/5, wood=this.params.preset==='wood'||this.params.preset==='crosswind';
    const key=[w,h,s.toFixed(2),this.params.preset,beauty,Boolean(shared)].join('/');
    if(this.cacheScene!==key){this.cacheScene=key;this.sceneCache.clear();}
    const cached=this.sceneCache.get(level);
    if(cached)return cached;
    const back=document.createElement('canvas');back.width=w;back.height=h;
    const c=back.getContext('2d',{alpha:false});
    c.fillStyle='#0c0e11'; c.fillRect(0,0,w,h);
    c.translate(cx,ground); c.scale(s,s);
    const tones=shared&&this.floorTones?.();
    c.fillStyle=grad(c,0,-.03,.88,tones||[[0,'#332b24'],[.47,'#211e1e'],[1,'#0c0e11']]);
    c.fillRect(-1.1,-1.3,2.2,1.8);
    if(beauty && !shared) {
      c.save();c.scale(1,.29);
      c.fillStyle=grad(c,0,0,.75,[[0,'rgba(155,75,30,'+(.54*glow)+')'],[.45,'rgba(89,46,27,'+(.43*glow)+')'],[1,'rgba(34,29,27,0)']]);
      c.fillRect(-.75,-.75,1.50,1.50);c.restore();
    }
    const rand=randomSequence(98152);
    for(let i=0;i<2700;i++) {
      const x=(rand()-.5)*2.8, z=rand()**2;
      const ry=.08-(1-z)**2*.20+z**4*.64;
      const x2=x*(.6+z), r=.0004+.0022*z*z;
      const light=1/(1+6*x2*x2+22*ry*ry);
      const warm=beauty?glow*light:0;
      c.fillStyle='rgba('+Math.round(85+65*warm)+','+Math.round(79+13*warm)+',68,'+(.04+rand()*.13)+')';
      c.fillRect(x2,ry,r*(1+1.4*rand()),r*.37+.00018);
    }
    if(wood) {
      // Bed of ash, fractured carbon, and glowing coals beneath the crossed logs.
      for(let i=0;i<380;i++) {
        const a=rand()*Math.PI*2, r=Math.sqrt(rand()), x=.29*r*Math.cos(a),y=.055*r*Math.sin(a)-.016;
        const sz=.0017+.014*rand()**2, hot=beauty&&glow>.025&&r<.85&&rand()<.24;
        c.fillStyle=hot?'rgba(201,76,23,'+(.82*glow)+')':'rgb('+[0,1,2].map((n)=>Math.floor(27+(2-n)*4+rand()*21)).join(',')+')';
        c.beginPath();c.moveTo(x-sz,y);c.lineTo(x-.3*sz,y-.5*sz);c.lineTo(x+.65*sz,y-.64*sz);c.lineTo(x+sz,y+.26*sz);c.lineTo(x-.50*sz,y+.40*sz);c.fill();
      }
      this.drawLog(c,-.227,-.015,.193,-.058,.032,17,glow);
      this.drawLog(c,-.16,-.083,.23,-.005,.029,39,glow);
    } else this.drawBurner(c,this.params.preset==='jet',glow);
    if(beauty && !shared && glow>.01) {
      c.save();c.scale(1,1.3);
      c.fillStyle=grad(c,0,-.115,.44,[[0,'rgba(232,82,13,'+(.13*glow)+')'],[.48,'rgba(150,61,18,'+(.075*glow)+')'],[1,'rgba(0,0,0,0)']]);
      c.fillRect(-.44,-.56,.88,.88);c.restore();
    }
    let front=null;
    if(wood) {
      front=document.createElement('canvas');front.width=Math.ceil(.66*s);front.height=Math.ceil(.16*s);
      const f=front.getContext('2d');f.translate(.33*s,.095*s);f.scale(s,s);
      this.drawLog(f,-.258,-.017,.141,.021,.027,7,glow);
      this.drawLog(f,.014,.025,.234,-.039,.021,54,glow);
    }
    const scene={back,front};this.sceneCache.set(level,scene);return scene;
  }
  draw(options={},output=this.canvas,c=this.ctx) {
    this.paintFire(options);
    const projection=this.projection(options,output),{s,ground,cx,gasHeight=s*1.12}=projection;
    const {back,front}=this.buildScene(options,projection);
    const shared=options.sharedLighting!==false && (!options.mode||options.mode==='beauty') && this.optics;
    c.drawImage(back,0,0);
    if(shared) shared.lightGround(c,projection,options.exposure??1);
    c.imageSmoothingEnabled=true;c.imageSmoothingQuality='low';
    const coalPower=front?this.radiantPower:(this.coalBedPower?.()??0);
    if(shared?.coals && (front || coalPower>0))shared.coals(c,projection,coalPower);
    c.drawImage(this.slice,cx-s*.5,ground-gasHeight,s,gasHeight);
    if(shared?.renderFlamelets)shared.renderFlamelets(c,projection,this.packets,options.exposure??1,options.quality);
    if(front){
      const layer=shared?shared.lightLogs(front,projection,options.exposure??1):front;
      c.drawImage(layer,Math.round(cx-.33*s),Math.round(ground-.095*s));
    }
    if(!options.mode||options.mode==='beauty') {
      c.lineCap='round';
      for(const e of this.sparks) {
        const flicker=.72+.28*Math.sin(e.seed+e.age*11);
        const alpha=clamp((1-e.age/e.life)*flicker,0,.91);
        c.strokeStyle='rgba(255,'+Math.round(132+94*alpha)+',66,'+alpha+')';
        c.lineWidth=Math.max(.62,.0014*s*e.size);
        const x=cx+(e.x/NX-.5)*s, y=ground-e.y/NY*gasHeight;
        c.beginPath();c.moveTo(x,y);c.lineTo(x+.0006*s,y+(.0015+e.size*.002)*s);c.stroke();
      }
    }
  }
  drawBurner(c,jet,glow) {
    const r=jet?.065:.31, side=jet?.071:.028, top=jet?-.073:-.018;
    c.fillStyle='#16191b'; c.beginPath(); c.ellipse(0,top+side,r,r*.155,0,0,Math.PI*2); c.fill();
    const g=c.createLinearGradient(-r,0,r,0);
    g.addColorStop(0,'#292b2b'); g.addColorStop(.26,'#383733'); g.addColorStop(.69,'#222323'); g.addColorStop(1,'#131415');
    c.fillStyle=g; c.fillRect(-r,top,2*r,side);
    c.fillStyle='#2d201a'; c.beginPath(); c.ellipse(0,top,r,r*.155,0,0,Math.PI*2); c.fill();
    c.strokeStyle='rgba(212,128,68,'+(.36+.3*glow)+')'; c.lineWidth=.0026; c.stroke();
  }
  drawLog(c,x0,y0,x1,y1,r,seed,glow) {
    const length=Math.hypot(x1-x0,y1-y0), rand=randomSequence(seed);
    c.save(); c.translate(x0,y0); c.rotate(Math.atan2(y1-y0,x1-x0));
    const grain=c.createLinearGradient(0,-r,0,r);
    grain.addColorStop(0,'#583428'); grain.addColorStop(.18,'#42251b'); grain.addColorStop(.52,'#251b18'); grain.addColorStop(1,'#100f10');
    c.fillStyle=grain; c.beginPath(); c.moveTo(0,-r);
    for(let i=1;i<=20;i++) c.lineTo(length*i/20,-r*(.84+.23*rand()));
    for(let i=20;i>=0;i--) c.lineTo(length*i/20,r*(.82+.19*rand())); c.closePath(); c.fill();
    for(let i=0;i<140;i++) {
      const x=.002+rand()*(length-.014), y=(rand()*1.68-.84)*r;
      const run=.003+rand()*.025, thick=.0006+rand()*.0038;
      const facing=clamp(.5-y/(r*2)+.16*(rand()-.5),0,1);
      const shade=Math.round(17+facing*22+rand()*14);
      c.fillStyle='rgb('+Math.round(shade+facing*18*glow)+','+Math.round(shade*.82)+','+Math.round(shade*.78)+')';
      c.beginPath(); c.moveTo(x,y);c.lineTo(x+.38*run,y-thick);c.lineTo(x+run,y-.7*thick);
      c.lineTo(x+.86*run,y+thick);c.lineTo(x+.12*run,y+.4*thick);c.fill();
    }
    for(let i=0;i<95;i++) {
      const x=rand()*length, y=(rand()*1.8-.94)*r, run=Math.min(length-x,.007+rand()*.06);
      const hot=glow>.025&&rand()<.17&&y<r*.2;
      c.strokeStyle=hot?'rgba(255,'+(77+Math.floor(90*rand()))+',26,'+(glow*(.2+.72*rand()))+')':
        rand()<.48?'rgba(12,12,13,.72)':'rgba(183,134,102,.27)';
      c.lineWidth=.00085+rand()*.00175; c.beginPath(); c.moveTo(x,y);
      c.lineTo(x+run*.49,y+rand()*.005-.0025); c.lineTo(x+run,y+rand()*.006-.003); c.stroke();
    }
    c.fillStyle='#2d211b'; c.beginPath(); c.ellipse(0,0,r*.42,r,0,0,Math.PI*2); c.fill();
    for(let j=1;j<=4;j++) {c.strokeStyle='rgba(139,86,54,.48)'; c.lineWidth=.0012; c.beginPath();c.ellipse(0,0,r*.39*j/4,r*.90*j/4,0,0,Math.PI*2);c.stroke();}
    c.restore();
  }
  destroy(){this.sparks.length=0;this.sceneCache.clear();}
}

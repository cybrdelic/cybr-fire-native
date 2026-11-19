// The planar solver uses the same temperature lookup as the volume renderer.
import {Fallback as OriginalFallback} from '../original/fallback.js';
import {createThermalTable} from './optics.js';
import {burstAmount,continuousSource,cpuSource,emberDraft,sourceShape} from './cpu-sources.js';

const nx=104,ny=124;
const unit=x=>Math.max(0,Math.min(1,x));
const twoPi=2*Math.PI;
const worldWidth=1.85, worldHeight=2.35;

function lookup(field,i,fx,fy,stride){
  const lo=field[i]+fx*(field[i+1]-field[i]);
  const hi=field[i+stride]+fx*(field[i+stride+1]-field[i+stride]);
  return lo+fy*(hi-lo);
}
const displayTable=Uint8Array.from({length:8193},(_,i)=>{
  // Shoulder for HDR radiance, followed by the sRGB transfer function.
  const x=i/256;
  const l=unit((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14));
  return Math.round(255*(l<=.0031308?12.92*l:1.055*Math.pow(l,1/2.4)-.055));
});
// Band-limited fluctuations at the inlet, sampled in transported material
// coordinates. No screen coordinates or frame-index random source enter here.
const eddies=new Float32Array(64*64);
let eddySeed=8171;
for(let i=0;i<eddies.length;i++){
  eddySeed=(Math.imul(eddySeed,1664525)+1013904223)>>>0;
  eddies[i]=eddySeed/4294967296;
}
function eddy(u,v){
  const x=Math.floor(u),y=Math.floor(v),a=x&63,b=(x+1)&63,lo=(y&63)*64,hi=((y+1)&63)*64;
  const dx=u-x,dy=v-y,fx=dx*dx*(3-2*dx),fy=dy*dy*(3-2*dy);
  const left=eddies[lo+a]+(eddies[lo+b]-eddies[lo+a])*fx;
  const right=eddies[hi+a]+(eddies[hi+b]-eddies[hi+a])*fx;
  return left+(right-left)*fy;
}
const display=r=>displayTable[Math.max(0,Math.min(8192,Math.round(r*256)))];

function shadeChannel(gas,light,ix,fx,lx,lf,emission,smoke){
  const e=gas[ix]+fx*(gas[ix+1]-gas[ix]);
  const l=light[lx]+lf*(light[lx+1]-light[lx]);
  return display(emission*e+smoke*l);
}

class SliceOptics {
  constructor(){
    this.thermal=createThermalTable();
    this.emissionFront=new Float32Array(nx*ny);
    this.lastTime=-1;
    this.rgb=Array.from({length:3},()=>new Float32Array(nx*ny));
    this.lamps=Array.from({length:3},()=>({x:0,y:0,r:0,g:0,b:0,weight:0}));
    this.sx=14;this.sy=16;
    this.gasRows=Array.from({length:3},()=>new Float64Array(nx));
    this.lightRows=Array.from({length:3},()=>new Float64Array(this.sx));
    this.smokeLight=Array.from({length:3},()=>new Float32Array(this.sx*this.sy));
  }

  update(state,burn,luminance,preset,time){
    if(time===this.lastTime)return;
    this.lastTime=time;
    const jet=preset==='jet'||preset==='pulsing';
    const pulse=preset==='pulsing';
    const burst=preset==='fireball'||preset==='groundburst';
    const wood=preset==='wood'||preset==='crosswind';
    const [red,green,blue]=this.rgb, table=this.thermal;
    const lamps=this.lamps;
    for(const lamp of lamps)Object.assign(lamp,{x:0,y:0,r:0,g:0,b:0,weight:0});
    for(let y=0,i=0;y<ny;y++)for(let x=0;x<nx;x++,i++){
      const t=state.heat[i];
      let front=0;
      if(luminance)front=luminance[i];
      else if(t>=.09){
        const k=Math.max(1,Math.min(ny-2,y))*nx+Math.max(1,Math.min(nx-2,x));
        const edge=Math.abs(state.heat[k+1]-state.heat[k-1])+Math.abs(state.heat[k+nx]-state.heat[k-nx]);
        const sheet=unit(edge*1.25+1.8*state.fuel[i]*state.air[i]);
        front=(t-.083)*(.54+.14*unit(burn[i]*.8)+.55*sheet);
      }
      if(front<=0){red[i]=green[i]=blue[i]=this.emissionFront[i]=0;continue;}
      // Resolve interleaved cold channels and hot sheets using the reacting
      // plume's own temperature peaks relative to its local thermal envelope.
      const east=i+Math.min(3,nx-1-x),west=i-Math.min(3,x);
      const north=i+Math.min(2,ny-1-y)*nx,south=i-Math.min(2,y)*nx;
      const neighborhood=(state.heat[east]+state.heat[west]+state.heat[north]+state.heat[south])*.25;
      const ratio=t/(.045+neighborhood);
      const ridge=unit((ratio-.61)/.59);
      const reaction=burn[i]/(.13+burn[i]);
      const m=state.filament[i],v=state.inletTime[i];
      // Two independent coordinates stretch and fold with the velocities.
      // Filter both Fourier bands when a grid cell covers their features.
      const jx=i+(x<nx-1?1:-1),jy=i+(y<ny-1?nx:-nx);
      const du=17*(Math.abs(m-state.filament[jx])+Math.abs(m-state.filament[jy]));
      const dv=3.2*(Math.abs(v-state.inletTime[jx])+Math.abs(v-state.inletTime[jy]));
      const footprint=Math.max(du,dv);
      const u=17*m+.7*v, w=3.2*v-1.4*m;
      const n=(eddy(u,w)-.5)*unit(1.55-footprint)
        +.42*(eddy(2.1*u+7,2.1*w+13)-.5)*unit(1.55-2.1*footprint);
      const fold=unit(.54+1.65*n);
      const thickness=(wood?.085:burst?.06:.11)+(.38+.52*reaction)*ridge*ridge*(.23+1.40*fold);
      this.emissionFront[i]=front*thickness;
      // Reduced heat is dimensionless. Map it to the calibrated visible
      // temperature range before looking up continuum power and chroma.
      const base=wood?1025:burst?1125:1190;
      const kelvin=base+((wood?745:655)+140*fold)*Math.sqrt(Math.min(2.2,t));
      const j=4*Math.round(255*unit((kelvin-650)/2000));
      const hotCarbon=unit(state.smoke[i]*2.8);
      const opacity=hotCarbon/(.35+hotCarbon);
      const radiance=(burst?.34:wood?1.11:pulse?.54:1.02)*table[j+3]*opacity*this.emissionFront[i]/(.17+t);
      const band=jet?Math.min(.52,.34*burn[i]*state.air[i]*state.air[i])*(pulse?Math.max(0,1-y/24):1):0;
      const r=red[i]=radiance*table[j]+.10*band;
      const g=green[i]=radiance*table[j+1]+.42*band;
      const b=blue[i]=radiance*table[j+2]+band;
      const lamp=lamps[Math.min(2,Math.floor(x/nx*3))];
      const l=.2126*r+.7152*g+.0722*b;
      lamp.x+=x*l;lamp.y+=y*l;lamp.weight+=l;
      lamp.r+=r;lamp.g+=g;lamp.b+=b;
    }
    for(const lamp of lamps){
      const norm=1/Math.max(lamp.weight,1e-8);
      lamp.x*=norm;lamp.y*=norm;
      const area=1/(nx*ny);
      lamp.r*=area;lamp.g*=area;lamp.b*=area;
    }
    this.computeSmokeLight(state.smoke);
  }

  computeSmokeLight(soot){
    const sx=this.sx, sy=this.sy;
    const [red,green,blue]=this.smokeLight;
    for(let y=0,i=0;y<sy;y++)for(let x=0;x<sx;x++,i++){
      const wx=x*(nx-1)/(sx-1),wy=y*(ny-1)/(sy-1);
      let r=0,g=0,b=0;
      for(const lamp of this.lamps){
        const dx=lamp.x-wx,dy=lamp.y-wy,dist=Math.hypot(dx,dy)/nx;
        let opticalDepth=0;
        for(let n=0;n<5;n++){
          const f=(n+.5)/5;
          const xx=Math.max(0,Math.min(nx-1.001,wx+f*dx));
          const yy=Math.max(0,Math.min(ny-1.001,wy+f*dy));
          const ix=xx|0, iy=yy|0;
          opticalDepth+=lookup(soot,iy*nx+ix,xx-ix,yy-iy,nx);
        }
        const flux=.10*Math.exp(-opticalDepth*dist*.95)/(.055+dist*dist);
        r+=flux*lamp.r;g+=flux*lamp.g;b+=flux*lamp.b;
      }
      red[i]=r;green[i]=g;blue[i]=b;
    }
  }

  prepareRow(iy,fy,sampleX,fractX,interpolateGas=true){
    if(this.columns!==sampleX){
      this.columns=sampleX;
      this.lightX=new Int16Array(sampleX.length);
      this.lightFx=new Float64Array(sampleX.length);
      for(let x=0;x<sampleX.length;x++){
        const gx=(sampleX[x]+fractX[x])*(this.sx-1)/(nx-1);
        this.lightX[x]=gx|0;this.lightFx[x]=gx-(gx|0);
      }
    }
    // Bilinear interpolation is separable. Complete the vertical component
    // once per row, then shade each pixel using two neighboring row samples.
    const gy=(iy+fy)*(this.sy-1)/(ny-1);
    const ky=gy|0, lightFy=gy-ky;
    for(let c=0;c<3;c++){
      const gas=this.rgb[c],g=this.gasRows[c];
      if(interpolateGas)for(let x=0,i=iy*nx;x<nx;x++,i++)g[x]=gas[i]+fy*(gas[i+nx]-gas[i]);
      const light=this.smokeLight[c],row=this.lightRows[c], ambient=.033+.008*c+.038*gy/this.sy;
      for(let x=0,i=ky*this.sx;x<this.sx;x++,i++)
        row[x]=ambient+.38*(light[i]+lightFy*(light[i+this.sx]-light[i]));
    }
  }

  shade(pixels,j,x,ix,fx,soot,f,fold,exposure){
    const lx=this.lightX[x],lf=this.lightFx[x];
    const smokeLevel=exposure*(1-f)/(1+.65*soot);
    // The folded front modulates emitting path length through this flat slice.
    const emission=exposure*f*fold*fold*fold;
    const gas=this.gasRows,light=this.lightRows;
    pixels[j]=shadeChannel(gas[0],light[0],ix,fx,lx,lf,emission,smokeLevel);
    pixels[j+1]=shadeChannel(gas[1],light[1],ix,fx,lx,lf,emission,smokeLevel);
    pixels[j+2]=shadeChannel(gas[2],light[2],ix,fx,lx,lf,emission,smokeLevel);
  }

  pool(c,x,y,radius,lamp,strength,exposure){
    const r=display(lamp.r*1.52*exposure),g=display(lamp.g*.83*exposure),b=display(lamp.b*.42*exposure);
    const grad=c.createRadialGradient(x,y,0,x,y,radius);
    const rgb=r+','+g+','+b+',';
    grad.addColorStop(0,'rgba('+rgb+strength+')');
    grad.addColorStop(.52,'rgba('+rgb+(strength*.24)+')');
    grad.addColorStop(1,'rgba('+rgb+'0)');
    c.fillStyle=grad;c.fillRect(x-radius,y-radius,2*radius,2*radius);
  }

  lightGround(c,{cx,ground,s},exposure){
    // Evaluate changing diffuse light at the floor's own spatial bandwidth,
    // then bilinearly reconstruct it on the bounded scene target.
    if(!this.ground){
      this.ground=document.createElement('canvas');
      this.ground.width=240;this.ground.height=64;
    }
    const canvas=this.ground,g=canvas.getContext('2d');
    g.clearRect(0,0,240,64);
    g.save();g.translate(120,29);g.scale(128,128*.30);
    g.globalCompositeOperation='screen';
    for(const lamp of this.lamps){
      const x=lamp.x/(nx-1)-.5, height=lamp.y/(ny-1);
      this.pool(g,x,-.04,.31+.64*height,lamp,.58/(1+height),exposure);
    }
    g.restore();
    c.save();c.globalCompositeOperation='screen';
    c.drawImage(canvas,cx-s*120/128,ground-s*29/128,s*240/128,s*64/128);
    c.restore();
  }

  renderFlamelets(ctx,projection,packets,exposure,quality){
    if(!packets.length)return;
    if(!this.flamelets){
      this.flamelets=[];
      for(let k=0;k<8;k++){
        const canvas=document.createElement('canvas');
        canvas.width=28;canvas.height=56;
        const c=canvas.getContext('2d'),img=c.createImageData(28,56);
        for(let y=0,j=0;y<56;y++)for(let x=0;x<28;x++,j+=4){
          const v=(y+.5)/56,u=(x+.5)/14-1;
          // Across a folded sheet the integrated path is longer near a tangent.
          // Apodize every edge so nearby packets can separate continuously.
          const mid=.55*Math.sin((v*1.18+k*.37)*twoPi)*(1-v)**.6 + .10*Math.sin((v*2.4+k*.11)*twoPi);
          const thickness=.045+.34*Math.sin(Math.PI*v)**.85;
          const d=(u-mid)/thickness,edge=Math.exp(-3.8*(Math.abs(d)-.66)**2);
          const vol=Math.exp(-1.7*d*d),envelope=Math.sin(Math.PI*v)**1.05;
          const amp=unit((.64*edge+.32*vol)*envelope);
          img.data[j]=245;img.data[j+1]=97+81*vol+22*v;
          img.data[j+2]=20+43*vol*v;img.data[j+3]=Math.round(225*amp);
        }
        c.putImageData(img,0,0);this.flamelets.push(canvas);
      }
    }
    const {cx,ground,s,gasHeight}=projection;
    ctx.save();ctx.globalCompositeOperation='screen';
    for(const p of packets){
      if(p.alpha<.005 || (quality==='low' && p.tile>2))continue;
      const length=gasHeight*p.length, width=s*p.width;
      const x=cx+(p.x/(nx-1)-.5)*s,y=ground-p.y/(ny-1)*gasHeight;
      const cs=Math.cos(p.angle),sn=Math.sin(p.angle);
      ctx.setTransform(cs,sn,-sn,cs,x,y);
      ctx.globalAlpha=.63*unit(exposure*p.alpha);
      ctx.drawImage(this.flamelets[p.tile],-width*.5,-length*.6,width,length);
    }
    ctx.restore();
  }

  coals(c,{cx,ground,s},power){
    if(!this.bed){
      const bed=this.bed=document.createElement('canvas');
      bed.width=320;bed.height=64;
      const ctx=bed.getContext('2d');
      let seed=631;
      const rand=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296);
      // Nearer pieces occlude the rear ones. Each coal retains a dark shell
      // around the exposed, internally glowing fracture.
      const pieces=Array.from({length:108},()=>{
        const angle=rand()*twoPi,rad=Math.sqrt(rand());
        return [160+151*rad*Math.cos(angle),31+21*rad*Math.sin(angle),rad,4+rand()**2*15,rand()];
      }).sort((a,b)=>a[1]-b[1]);
      for(const [x,y,rad,w,n] of pieces){
        const h=w*(.25+.29*rand());
        ctx.fillStyle='rgb('+[36+20*n,20+8*n,15+8*n].map(Math.round).join(',')+')';
        ctx.beginPath();ctx.moveTo(x-w,y);ctx.lineTo(x-w*.66,y-.8*h);ctx.lineTo(x+.2*w,y-h);
        ctx.lineTo(x+.92*w,y-.5*h);ctx.lineTo(x+w,y+h*.24);ctx.lineTo(x+.22*w,y+.75*h);
        ctx.lineTo(x-w*.72,y+.52*h);ctx.fill();
        if(rad>.87||n<.22)continue;
        const x0=x-.78*w,y0=y-.38*h,dx=w*(.65+.75*rand()),jitter=(rand()-.5)*h*.6;
        const r=169+70*n,g=36+93*n;
        const crack=(width,color,run)=>{
          ctx.lineWidth=width;ctx.strokeStyle=color;
          ctx.beginPath();ctx.moveTo(x0,y0);ctx.lineTo(x0+.29*dx,y0+.27*h);
          ctx.lineTo(x0+.57*dx,y0+jitter);ctx.lineTo(x0+run*dx,y0-.2*h);ctx.stroke();
        };
        crack(1.9+.075*w,'rgb('+Math.round(r)+','+Math.round(g)+',17)',1);
        if(n>.72)crack(.8,'#ffbc53',.77);
      }
    }
    c.save();c.globalAlpha=unit(power*1.45);
    c.drawImage(this.bed,cx-.295*s,ground-.063*s,.59*s,.108*s);
    c.restore();
  }

  lightLogs(front,{s},exposure){
    this.front??=document.createElement('canvas');
    const layer=this.front;
    if(layer.width!==front.width)layer.width=front.width;
    if(layer.height!==front.height)layer.height=front.height;
    const ctx=layer.getContext('2d');
    ctx.clearRect(0,0,layer.width,layer.height);ctx.drawImage(front,0,0);
    ctx.save();ctx.translate(.33*s,.095*s);ctx.scale(s,s);
    ctx.globalCompositeOperation='source-atop';
    for(const lamp of this.lamps){
      const height=lamp.y/(ny-1);
      this.pool(ctx,lamp.x/(nx-1)-.5,-.10,.23+.60*height,lamp,.25/(1+height),exposure);
    }
    ctx.restore();
    return layer;
  }
}

export class Fallback extends OriginalFallback {
  constructor(canvas){
    super(canvas);
    this.optics=new SliceOptics();
    this.coverage=Float32Array.from({length:257},(_,i)=>{
      // Integrate a sinusoid over a box footprint. Suppress lobes beyond Nyquist.
      const x=i*Math.PI/256;
      return i?Math.sin(x)/x:1;
    });
  }

  reset(params={}){
    super.reset(params);
    if(this.optics)this.optics.lastTime=-1;
    this.a.inletTime??=new Float32Array(nx*ny);
    this.b.inletTime??=new Float32Array(nx*ny);
    for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)this.a.inletTime[x+y*nx]=-y/43+.06*Math.sin(x*.41);
    this.paintKey=null;
    this.inlet=null;
    this.viewKey=null;
    this.packets=[];
    this.packetBudget=0;
    this.packetSeed=39517;
    this.customSource=cpuSource(this.params.preset);
    this.customGains=null;
    if(this.customSource){
      this.customGains=new Float32Array(nx);
      this.source.fill(0);
      this.burn.fill(0);
      this.radiantPower=0;
      this.sparks.length=0;
      for(const field of ['heat','smoke','fuel','u','v'])this.a[field].fill(0);
      this.a.air.fill(this.params.oxygen);
      if(!this.customSource.spread){
        for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
          const wx=(x/(nx-1)-.5)*worldWidth,wy=y/(ny-1)*worldHeight;
          this.source[x+y*nx]=sourceShape(this.params.preset,this.customSource,this.params.preset==='pulsing'?wx*1.65:wx,wy);
        }
      }
    }
  }

  advance(dt){
    const custom=this.customSource;
    const dose=this.params.coldStart?0:burstAmount(custom,this.time-dt,dt);
    if(dose>0)this.depositBurst(custom,dose);
    super.advance(dt);
    if(custom?.spread){
      // The fast front has a finite thermal lifetime. The reduced slice
      // entrains cooler gas from its missing depth dimension.
      const cool=1-dt*.91;
      for(let i=0;i<this.a.heat.length;i++)this.a.heat[i]*=cool;
    }
  }

  depositBurst(profile,dose){
    const {a,params}=this;
    const sweeping=params.preset==='groundburst';
    const strength=dose*params.fuel*Math.min(1.2,1.45*params.oxygen);
    const minX=Math.max(1,Math.floor((.5+(profile.x-1.4*profile.radius)/worldWidth)*(nx-1)));
    const maxX=Math.min(nx-1,Math.ceil((.5+(profile.x+1.4*profile.radius)/worldWidth)*(nx-1)));
    const minY=Math.max(1,Math.floor((profile.y-1.4*profile.height)/worldHeight*(ny-1)));
    const maxY=Math.min(ny-1,Math.ceil((profile.y+1.4*profile.height)/worldHeight*(ny-1)));
    for(let y=minY;y<maxY;y++)for(let x=minX;x<maxX;x++){
      const wx=(x/(nx-1)-.5)*worldWidth,wy=y/(ny-1)*worldHeight;
      const mass=strength*sourceShape(params.preset,profile,wx,wy);
      if(mass<=0)continue;
      const i=x+y*nx, dx=(wx-profile.x)/profile.radius,dy=(wy-profile.y)/Math.max(.23,profile.height);
      a.heat[i]+=1.75*mass*profile.heat;
      a.fuel[i]+=1.44*mass*profile.fuel;
      a.smoke[i]+=mass*profile.soot;
      a.u[i]+=mass*(profile.spread*dx+(sweeping?2.40:0))*(nx-1)/worldWidth;
      a.v[i]+=mass*(profile.lift+1.5*dy)*(ny-1)/worldHeight;
    }
  }

  packetRandom(){
    this.packetSeed=(Math.imul(this.packetSeed,1664525)+1013904223)>>>0;
    return this.packetSeed/4294967296;
  }

  step(dt,params={},time){
    super.step(dt,params,time);
    if(!(dt>0))return;
    if(this.customSource)return;
    dt=Math.min(.065,dt);
    const a=this.a, jet=this.params.preset==='jet';
    // A planar slice averages many individual out-of-plane flame sheets.
    // Seed small luminous gas packets only from hot, reacting inlet cells.
    this.packetBudget+=dt*this.radiantPower*(jet?45:82);
    while(this.packetBudget>=1){
      this.packetBudget--;
      const x=nx*(.5+(this.packetRandom()-.5)*(jet?.13:.51)),y=5+this.packetRandom()*7;
      const i=(x|0)+(y|0)*nx;
      if(a.heat[i]<.32||this.burn[i]<.04)continue;
      const w=this.packetRandom();
      this.packets.push({x,y,angle:0,age:0,alpha:0,speed:.80+this.packetRandom()*.37,
        width:.025+.054*w,length:.055+.12*w,life:.42+.85*this.packetRandom(),
        heat:.26+.44*this.packetRandom(),tile:Math.floor(8*this.packetRandom())});
    }
    for(let k=this.packets.length-1;k>=0;k--){
      const p=this.packets[k],ix=p.x|0,iy=p.y|0,i=iy*nx+ix,fx=p.x-ix,fy=p.y-iy;
      if(ix<1||ix>=nx-2||iy<1||iy>=ny-2||p.age>=p.life){this.packets.splice(k,1);continue;}
      const u=lookup(a.u,i,fx,fy,nx),v=Math.max(2,lookup(a.v,i,fx,fy,nx));
      const t=lookup(a.heat,i,fx,fy,nx);
      p.x+=dt*u*p.speed;p.y+=dt*v*p.speed;
      p.angle+=Math.min(1,dt*9)*(Math.atan2(u,v)-p.angle);
      p.age+=dt;
      const residence=unit(p.age/p.life),edge=4*residence*(1-residence);
      p.alpha=p.heat*edge*edge*unit((t-.06)*1.8);
    }
  }

  burnerGains(params,time){
    if(this.customGains){
      const pulse=continuousSource(params.preset,time);
      if(!pulse)return null;
      const embers=params.preset==='embers';
      const drive=params.coldStart?0:(embers?.74:.40)*pulse.feed;
      for(let x=0;x<nx;x++){
        this.customGains[x]=drive*(embers?emberDraft((x/(nx-1)-.5)*worldWidth,0,time):1);
      }
      return this.customGains;
    }
    if(params.expansion===false || (params.preset!=='wood' && params.preset!=='crosswind'))return null;
    if(!this.inlet){
      const vents=[.263,.355,.458,.566,.70], widths=[.023,.030,.026,.032,.030];
      this.inlet=vents.map((center,k)=>Float32Array.from({length:nx},(_,x)=>
        Math.exp(-.5*((x/(nx-1)-center)/widths[k])**2)));
      this.gains=new Float32Array(nx);
      this.supply=Array.from({length:nx},(_,x)=>this.source[4*nx+x]);
      this.supplyTotal=this.supply.reduce((sum,v)=>sum+v,0);
    }
    // Volatiles leave separate cracks in the timber. Their independent
    // release histories form distinct shear layers which the solver advects.
    this.gains.fill(.085);
    let total=0;
    for(let k=0;k<this.inlet.length;k++){
      const slow=Math.sin(time*(2.13+.27*k)+k*4.7);
      const fast=Math.sin(time*(6.60-.29*k)-k*2.9);
      const drive=.28+1.22*unit((slow+.37*fast+.44)*.75)**1.5;
      for(let x=0;x<nx;x++)this.gains[x]+=drive*this.inlet[k][x];
    }
    for(let x=0;x<nx;x++)total+=this.supply[x]*this.gains[x];
    const normalize=this.supplyTotal/(total||1);
    for(let x=0;x<nx;x++)this.gains[x]*=normalize;
    return this.gains;
  }

  applyThermalExpansion(divergence,heat,burn,dt,state){
    if(this.params.expansion===false)return;
    const ox=this.params.oxygen, jet=this.params.preset==='jet'||this.reducedNozzle();
    for(let y=1;y<ny-1;y++){
      const h=y/(ny-1),rise=unit((h-.10)/.65);
      // The missing depth axis admits fresh ambient gas along the widening
      // plume. Mix species and enthalpy consistently before the next step.
      const mix=dt*(jet?.72:1.65)*rise*rise*(3-2*rise);
      for(let i=y*nx+1,end=(y+1)*nx-1;i<end;i++){
        divergence[i]-=Math.min(1.6,1.55*burn[i]/(1.65+heat[i]));
        heat[i]*=1-mix;state.fuel[i]*=1-mix;state.smoke[i]*=1-mix*.62;
        state.air[i]+=mix*(ox-state.air[i]);
      }
    }
  }

  projection(options={},output=this.canvas){
    if(options.sharedLighting===false&&options.stableSampling===false&&this.params.expansion===false)
      return super.projection(options,output);
    const w=output.width,h=output.height, zoom=Math.max(.62,Math.min(1.8,3.9/(options.distance||3.9)));
    const size=Math.min(w*1.03,h*1.04);
    const gasHeight=Math.min(h*.72,size*.87)*zoom;
    const shift=((options.targetHeight??.98)-.98)*gasHeight/worldHeight;
    return {w,h,cx:w*.5,ground:Math.max(.30*h,Math.min(.92*h,h*.84+shift)),s:size*zoom,gasHeight};
  }

  drawBurner(ctx,nozzle,glow){
    const preset=this.params.preset;
    if(preset==='fireball'||preset==='groundburst'||preset==='embers')return;
    super.drawBurner(ctx,nozzle||preset==='pulsing',glow);
  }

  reducedNozzle(){
    return this.params.preset==='pulsing';
  }

  coalBedPower(){
    return this.params.preset==='embers'?.39*this.params.fuel+this.radiantPower*.72:0;
  }

  floorTones(){
    return [[0,'#282723'],[.47,'#1c1d1e'],[1,'#0c0e11']];
  }

  paintFire(options){
    const key=[this.time,options.mode,options.quality,options.exposure,options.stableSampling,options.sharedLighting].join('/');
    if(this.paintKey===key)return;
    if((!options.mode||options.mode==='beauty') && options.stableSampling!==false &&
      options.sharedLighting!==false && (options.quality==='low'||options.quality==='medium')){
      this.paintResolved(options);
    }else super.paintFire(options);
    this.paintKey=key;
  }

  paintResolved(options){
    // The thermochemical solution lives on the solver grid. At the two
    // economical tiers shade its samples once, then reconstruct radiance with
    // hardware bilinear filtering. The optical footprint is explicit.
    this.prepareSlice('low');
    const {a,burn,optics,opacity}=this, pixels=this.image.data;
    optics.update(a,burn,null,this.params.preset,this.time);
    const [r,g,b]=optics.rgb,[lr,lg,lb]=optics.lightRows;
    const exp=Math.max(.12,Math.min(4,options.exposure??1));
    for(let y=0;y<ny;y++){
      const iy=Math.min(y,ny-2),fy=y-iy;
      optics.prepareRow(iy,fy,this.sampleX,this.fractX,false);
      const top=unit((ny-1-y)/15), row=iy*nx;
      for(let x=0,i=y*nx,j=(ny-1-y)*nx*4;x<nx;x++,i++,j+=4){
        const front=optics.emissionFront[i],soot=a.smoke[i];
        if(front<.0006 && soot<.0018){pixels[j+3]=0;continue;}
        const ix=Math.min(x,nx-2),fold=front>.002?this.filterFold(a.filament,row+ix,x-ix,fy,y,1):1;
        const flame=opacity[Math.min(2048,front*fold*1485)|0];
        const smoke=opacity[Math.min(2048,soot*435)|0]*(1-flame),alpha=flame+smoke;
        if(alpha<.003){pixels[j+3]=0;continue;}
        const em=exp*flame*fold*fold*fold/alpha,sc=exp*smoke/(alpha*(1+.65*soot));
        const k=optics.lightX[x],f=optics.lightFx[x];
        pixels[j]=display(em*r[i]+sc*(lr[k]+f*(lr[k+1]-lr[k])));
        pixels[j+1]=display(em*g[i]+sc*(lg[k]+f*(lg[k+1]-lg[k])));
        pixels[j+2]=display(em*b[i]+sc*(lb[k]+f*(lb[k+1]-lb[k])));
        pixels[j+3]=255*alpha*top*unit(Math.min(x,nx-1-x)/8);
      }
    }
    this.sliceContext.putImageData(this.image,0,0);
  }

  draw(options={}){
    const canvas=this.canvas,w=canvas.width,h=canvas.height;
    const key=[this.time,w,h,options.mode,options.quality,options.exposure,options.distance,options.targetHeight,
      options.stableSampling,options.sharedLighting].join('/');
    if(key===this.viewKey)return;
    this.viewKey=key;
    const limit={low:86000,medium:224000,high:375000,ultra:540000}[options.quality]||224000;
    const scale=Math.min(1,Math.sqrt(limit/(w*h)));
    if(scale>=1 || (options.sharedLighting===false&&options.stableSampling===false&&this.params.expansion===false)){
      super.draw(options);
      return;
    }
    this.stage??=document.createElement('canvas');
    const target=this.stage, width=Math.max(1,Math.round(w*scale)),height=Math.max(1,Math.round(h*scale));
    if(target.width!==width)target.width=width;
    if(target.height!==height)target.height=height;
    const ctx=target.getContext('2d',{alpha:false});
    super.draw(options,target,ctx);
    this.ctx.imageSmoothingEnabled=true;
    this.ctx.drawImage(target,0,0,w,h);
  }

  filterFold(mat,i,fx,fy,sy,scale){
    const m=lookup(mat,i,fx,fy,nx),ph=this.phase;
    const dx=((mat[i+1]-mat[i])*(1-fy)+(mat[i+nx+1]-mat[i+nx])*fy)/scale;
    const dy=((mat[i+nx]-mat[i])*(1-fx)+(mat[i+nx+1]-mat[i+1])*fx)/scale;
    const wave=m*3680+sy*8;
    const warp=ph[(wave|0)&1023], slope=112*twoPi/1024*ph[((wave+256)|0)&1023];
    const p=m*6973+2.4*sy+112*warp;
    const gradX=dx*(6973+3680*slope);
    const gradY=dy*(6973+3680*slope)+(2.4+8*slope)/scale;
    const footprint=Math.abs(gradX)+Math.abs(gradY);
    const fineFootprint=Math.abs(2.71*gradX+3120*dx)+Math.abs(2.71*gradY+3120*dy);
    const broad=ph[(p|0)&1023]*this.coverage[Math.min(256,Math.round(footprint*.25))];
    const fine=ph[((2.71*p+3120*m)|0)&1023]*this.coverage[Math.min(256,Math.round(fineFootprint*.25))];
    const ridge=unit((broad+.25*fine+.43)*1.2);
    // Resolve the folded sheet into bright filaments and translucent gaps.
    // The footprint already removes unresolved folds at minification.
    return .67+.56*ridge+.075*fine*ridge;
  }
}

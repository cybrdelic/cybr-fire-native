import {burstAmount,continuousSource,cpuSource,emberDraft,sourceShape} from './cpu-sources.js';

// Collocated 3D flow on the CPU. Distances are world units, velocity is
// world units per second, and the chemical fields are relative quantities.
export const grid = {x:26,y:41,z:26,width:1.85,height:2.35};
export const wood = [
    [-.37,.085,-.23, .26,.088,.22,.069],
    [-.36,.074,.21, .33,.078,-.20,.072],
    [-.28,.18,-.22, .19,.18,.24,.062],
    [.34,.167,-.12, -.22,.17,.24,.063]
];
export const clamp = (v,lo,hi)=>Math.max(lo,Math.min(v,hi));
const {x:nx,y:ny,z:nz,width,height}=grid;
const stride=nx*ny, count=stride*nz;
const sx=(nx-1)/width, sy=(ny-1)/height;
const ax=sx*sx, ay=sy*sy, poissonWeight=1/(4*ax+2*ay);
const field=()=>new Float32Array(count);
const gas=()=>Array.from({length:6},field);

export function interpolate(v,i,x,y,z){
    const a=v[i]+(v[i+1]-v[i])*x;
    const b=v[i+nx]+(v[i+nx+1]-v[i+nx])*x;
    i+=stride;
    const c=v[i]+(v[i+1]-v[i])*x;
    const d=v[i+nx]+(v[i+nx+1]-v[i+nx])*x;
    const bottom=a+(b-a)*y;
    return bottom+(c+(d-c)*y-bottom)*z;
}

function burner(preset,x,y,z){
    if(preset==='pool'||preset==='jet'){
        const radius=preset==='jet'?.075:.255;
        return Math.exp(-(x*x+z*z)/(radius*radius)-((y-.09)/.074)**2);
    }
    let source=0;
    for(let k=0;k<wood.length;k++){
        const [x0,y0,z0,x1,,z1,r]=wood[k];
        const dx=x1-x0,dz=z1-z0;
        const u=clamp(((x-x0)*dx+(z-z0)*dz)/(dx*dx+dz*dz),0,1);
        const distance=(x-x0-u*dx)**2+(z-z0-u*dz)**2+1.2*(y-y0-.06)**2;
        source=Math.max(source,Math.exp(-distance/(1.4*r*r))*(.65+.35*Math.cos(11*u+1.93*k)));
    }
    return source;
}

function wave(table,k,angle){
    for(let i=0;i<table.length;i++)table[i]=Math.sin(k*i+angle);
}

export class CpuFlow {
    constructor(){
        this.dims=[nx,ny,nz];
        this.gas=gas();
        this.next=gas();
        this.pressure=field();
        this.divergence=field();
        this.source=field();
        this.sourcePhase=field();
        this.expansion=field();
        this.waves=[nx,nx,ny,ny,nz,nz].map(n=>new Float32Array(n));
    }
    reset(params){
        this.preset=params.preset||'wood';
        this.customSource=cpuSource(this.preset);
        this.cold=Boolean(params.coldStart);
        this.time=0;
        for(const f of [...this.gas,...this.next,this.pressure,this.divergence,this.expansion])f.fill(0);
        const [heat,fuel,smoke,,rise]=this.gas;
        for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1;x<nx-1;x++){
            const i=x+y*nx+z*stride;
            const px=x/sx-width*.5,py=y/sy,pz=z/sx-width*.5;
            this.source[i]=this.customSource?
                sourceShape(this.preset,this.customSource,px,py,pz):burner(this.preset,px,py,pz);
            this.sourcePhase[i]=3*Math.sin(9*px+4*pz)+2*Math.cos(13*pz-3*px);
            if(this.cold || this.customSource)continue;
            const radius=Math.max(.035,(this.preset==='jet'?.135:.28)-.14*py);
            const r2=(px-.045*Math.sin(py*4))**2+(pz-.04*Math.cos(py*5))**2;
            const active=Math.exp(-r2/(radius*radius)-.80*py)*Math.sqrt((params.fuel??.75)/.75);
            heat[i]=active*.83;fuel[i]=active*.84;smoke[i]=active*.13;rise[i]=active*.9;
        }
    }
    step(dt,params,time){
        dt=clamp(dt,0,1/15);
        if(!dt)return;
        this.time=time+dt;
        const [heat,fuel,soot,u,v,w]=this.gas;
        const [newHeat,newFuel,newSoot,newU,newV,newW]=this.next;
        const [x1,x2,y1,y2,z1,z2]=this.waves;
        wave(x1,.34,time*1.17+.3);wave(x2,.74,-time*2.14+1.4);
        wave(y1,.38,-time*2.77+1.4);wave(y2,.63,-time*3.13);
        wave(z1,.48,time*1.37+1);wave(z2,.74,time*2.6+2.2);
        const jet=this.preset==='jet', oxygen=clamp(params.oxygen??.85,0,1.5);
        const supply=clamp(params.fuel??.75,0,1.5);
        const custom=this.customSource, pulse=continuousSource(this.preset,time);
        const hotLoad=supply*Math.min(1.2,oxygen*1.45);
        const dose=this.cold?0:burstAmount(custom,time,dt);
        const finite=Boolean(custom?.spread), ember=this.preset==='embers', sweep=this.preset==='groundburst';
        const brush=params.brush;
        const bx=brush?brush[0]*(nx-1):0,by=brush?brush[1]*(ny-1):0,bz=brush?brush[2]*(nz-1):0;
        const curl=1.3+2*(params.turbulence??.64),wind=(params.wind??.04)*1.9;
        for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1;x<nx-1;x++){
            const i=x+y*nx+z*stride,source=this.source[i];
            const inlet=finite?(dose>0?source:0):source;
            const brushDistance=brush?(x-bx)**2+(y-by)**2+(z-bz)**2:100;
            const neighborSmoke=custom?.spread||ember?(soot[i-1]+soot[i+1]+soot[i-nx]+soot[i+nx]+soot[i-stride]+soot[i+stride])*.4:0;
            if(brushDistance>=12 && inlet<.00002 && heat[i]+soot[i]+fuel[i]+heat[i-1]+heat[i+1]+heat[i-nx]+heat[i+nx]+heat[i-stride]+heat[i+stride]+neighborSmoke<.002){
                newHeat[i]=newFuel[i]=newSoot[i]=this.expansion[i]=0;
                newU[i]=u[i]*.965;newV[i]=v[i]*.96;newW[i]=w[i]*.965;
                continue;
            }
            const px=clamp(x-u[i]*dt*sx,.01,nx-1.01),py=clamp(y-v[i]*dt*sy,.01,ny-1.01);
            const pz=clamp(z-w[i]*dt*sx,.01,nz-1.01),ix=px|0,iy=py|0,iz=pz|0;
            const j=ix+iy*nx+iz*stride,fx=px-ix,fy=py-iy,fz=pz-iz;
            let h=interpolate(heat,j,fx,fy,fz),f=interpolate(fuel,j,fx,fy,fz);
            const smoke=interpolate(soot,j,fx,fy,fz);
            const phase=this.sourcePhase[i];
            let feed=inlet*(1+.29*Math.sin(4.7*time+phase)+.12*Math.sin(8*time-.77*phase));
            let pushX=0,pushY=0,pushZ=0;
            if(finite){
                const packet=dose*source*hotLoad;
                h+=1.95*packet*custom.heat;f+=1.42*packet*custom.fuel;
                const dx=x/sx-width*.5-custom.x,dy=y/sy-custom.y,dz=z/sx-width*.5;
                pushX=packet*(custom.spread*dx/custom.radius+(sweep?3.10:0));
                pushY=packet*(custom.lift+1.8*dy/Math.max(.23,custom.height));
                pushZ=packet*custom.spread*dz/custom.radius;
                feed=0;
            }else if(pulse){
                const draft=ember?emberDraft(x/sx-width*.5,z/sx-width*.5,time):1;
                feed*=pulse.feed*draft;
                f+=dt*feed*supply*7;
                h+=this.cold?0:dt*feed*3.7*pulse.heat*hotLoad;
                pushY=dt*pulse.lift*feed;
            }else{
                f+=dt*feed*supply*7;
                h+=this.cold?0:dt*feed*3.7*clamp(supply/.75,0,1)*Math.min(1,oxygen/.55);
            }
            if(brushDistance<12){
                const ignition=1-brushDistance/12;
                h+=4*dt*ignition;f+=.9*dt*ignition;
            }
            const air=oxygen/(1+3*f);
            const burned=Math.min(f,dt*(1.52*Math.max(0,h-.08)*air+.035*f*h*oxygen)*(finite?2.35:1));
            f=Math.min(1.8,f-burned);
            h=clamp(h+1.7*burned-dt*(.04+(finite?1.08:.50)*h),0,finite?1.85:1.52);
            newHeat[i]=h;newFuel[i]=f;
            const fallout=finite?dose*source*hotLoad*custom.soot:0;
            const sootInlet=pulse?pulse.soot*feed:finite?0:source;
            newSoot[i]=Math.max(0,(smoke+fallout+burned*(jet?.05:finite?.37:.205)+dt*(ember?.11:.045)*sootInlet*supply)*(1-dt*.52));
            this.expansion[i]=params.expansion===false?0:Math.min(1.2,burned/dt*.42);
            const eddy=Math.min(1,2.15*h)*curl;
            newU[i]=interpolate(u,j,fx,fy,fz)+pushX+dt*(eddy*(.79*y1[y]*z1[z]+.44*y2[y]*z2[z])+.9*(wind-u[i]));
            newW[i]=interpolate(w,j,fx,fy,fz)+pushZ+dt*(eddy*(-.83*y1[y]*x1[x]+.47*y2[y]*x2[x])-.9*w[i]);
            const lift=jet?6.2:ember?2.3:4.3;
            newV[i]=Math.max(-.27,interpolate(v,j,fx,fy,fz)+pushY+dt*(h*lift-1.05*v[i]+.33*eddy*x2[x]*z2[z]));
        }
        this.project(newU,newV,newW);
        [this.gas,this.next]=[this.next,this.gas];
    }
    project(u,v,w){
        const p=this.pressure,div=this.divergence;
        for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1,i=1+y*nx+z*stride;x<nx-1;x++,i++){
            div[i]=.5*(sx*(u[i+1]-u[i-1]+w[i+stride]-w[i-stride])+sy*(v[i+nx]-v[i-nx]))-this.expansion[i];
        }
        // Warm-started pressure relaxation. The rectangular grid uses
        // different coefficients for its vertical and horizontal spacings.
        for(let k=0;k<4;k++)for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1,i=1+y*nx+z*stride;x<nx-1;x++,i++){
            p[i]=(ax*(p[i-1]+p[i+1]+p[i-stride]+p[i+stride])+ay*(p[i-nx]+p[i+nx])-div[i])*poissonWeight;
        }
        for(let z=1;z<nz-1;z++)for(let y=1;y<ny-1;y++)for(let x=1,i=1+y*nx+z*stride;x<nx-1;x++,i++){
            u[i]-=.5*sx*(p[i+1]-p[i-1]);v[i]-=.5*sy*(p[i+nx]-p[i-nx]);w[i]-=.5*sx*(p[i+stride]-p[i-stride]);
        }
    }
}

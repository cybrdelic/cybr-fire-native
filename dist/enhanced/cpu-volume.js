import {CpuFlow,grid,wood,clamp,interpolate} from './cpu-flow.js';

const {x:nx,y:ny,z:nz,width:domain,height:ceiling}=grid;
const stride=nx*ny,fxCount=(nx-1)*2,fyCount=(ny-1)*2,fzCount=(nz-1)*2;
const lens=.32;
const pixelBudgets={low:41000,medium:59000,high:84000,ultra:112000};
const tone=Uint8Array.from({length:16384},(_,i)=>Math.round(255*Math.pow(1-Math.exp(-i/1800),1/2.2)));
const logs=wood.map(([x,y,z,endX,endY,endZ,r],id)=>{
    const dx=endX-x,dy=endY-y,dz=endZ-z,length=Math.hypot(dx,dy,dz);
    return {x,y,z,r,id,length,dx:dx/length,dy:dy/length,dz:dz/length};
});

function camera({yaw=.38,pitch=.14,distance=3.9,targetHeight=.98}={}){
    const s=Math.sin(yaw),c=Math.cos(yaw),a=Math.sin(pitch),b=Math.cos(pitch);
    const ahead=[-s*b,-a,-c*b],right=[c,0,-s],up=[-s*a,b,-c*a];
    const origin=[-distance*ahead[0],targetHeight-distance*ahead[1],-distance*ahead[2]];
    return {ahead,right,up,origin};
}

function ambientLut(value){
    return tone[clamp((value*1800)|0,0,tone.length-1)];
}

// Integral of a smooth compact voxel footprint in the image. Adjacent
// reconstructed volume samples overlap and carry energy proportional to
// projected area; the kernels are shared across frame and camera changes.
function footprintKernels(width){
    const result=[];
    for(let half=2;half<=14;half++)for(let iy=0;iy<4;iy++)for(let ix=0;ix<4;ix++){
        const radius=half*.5,R=Math.ceil(radius),offsets=[],weights=[],xs=[],ys=[];
        let sum=0;
        for(let y=-R;y<=R;y++)for(let x=-R;x<=R;x++){
            const dx=x-(ix+.5)*.25,dy=y-(iy+.5)*.25;
            const q=1-(dx*dx+dy*dy)/(radius*radius);
            if(q<=0)continue;
            offsets.push(y*width+x);weights.push(q*q);xs.push(x);ys.push(y);sum+=q*q;
        }
        result[(half<<4)+(iy<<2)+ix]={offsets,weights:Float32Array.from(weights,w=>w/sum),xs,ys};
    }
    return result;
}

function material(ray,distance,log,hit){
    const x=ray.ox-log.x,y=ray.oy-log.y,z=ray.oz-log.z;
    const along=x*log.dx+y*log.dy+z*log.dz,travel=ray.x*log.dx+ray.y*log.dy+ray.z*log.dz;
    const ux=ray.x-travel*log.dx,uy=ray.y-travel*log.dy,uz=ray.z-travel*log.dz;
    const px=x-along*log.dx,py=y-along*log.dy,pz=z-along*log.dz;
    const a=ux*ux+uy*uy+uz*uz,b=ux*px+uy*py+uz*pz,c=px*px+py*py+pz*pz-log.r*log.r;
    const discriminant=b*b-a*c;
    if(discriminant<0||a<.000001)return distance;
    const side=(-b-Math.sqrt(discriminant))/a;
    let t=side,axial=along+t*travel,cap=false;
    if(t<=0||axial<0||axial>log.length){
        if(Math.abs(travel)<.00001)return distance;
        const axialPlane=travel>0?0:log.length;
        t=(axialPlane-along)/travel;
        if(t<=0||t>=distance)return distance;
        const ax=px+t*ux,ay=py+t*uy,az=pz+t*uz;
        if(ax*ax+ay*ay+az*az>log.r*log.r)return distance;
        axial=axialPlane;cap=true;
    }
    if(t>=distance||t<=0)return distance;
    const dx=px+t*ux,dy=py+t*uy,dz=pz+t*uz;
    const angle=Math.atan2(dy,dx*log.dz-dz*log.dx);
    const grain=cap?Math.hypot(dx,dy,dz)*74:angle*13+1.4*Math.sin(axial*22+log.id);
    const ridge=.40+.36*Math.cos(grain)+.12*Math.cos(grain*2.14+3);
    const lit=.34+.66*Math.max(0,dy/log.r);
    const ember=Math.max(0,Math.sin(angle*17+1.7*Math.sin(axial*15))-.85)*Math.max(0,Math.cos(axial*9+log.id));
    hit[0]=.055*ridge*lit+.64*ember;
    hit[1]=.023*ridge*lit+.13*ember;
    hit[2]=.010*ridge*lit+.012*ember;
    return t;
}

function coalCrack(x,z){
    const cellX=Math.floor(x),cellZ=Math.floor(z);
    let nearest=1e4,second=1e4,variant=0;
    for(let dz=-1;dz<=1;dz++)for(let dx=-1;dx<=1;dx++){
        const i=cellX+dx,j=cellZ+dz;
        let bits=Math.imul(i,17887)^Math.imul(j,62927);
        bits=Math.imul(bits^(bits>>>16),2246822519);
        bits=Math.imul(bits^(bits>>>13),3266489917);
        const px=i+.08+.84*(bits&65535)/65535;
        const pz=j+.08+.84*(bits>>>16)/65535;
        const d=(x-px)**2+(z-pz)**2;
        if(d<nearest){second=nearest;nearest=d;variant=(bits>>>11&511)/511;}
        else if(d<second)second=d;
    }
    const gap=clamp(1-6*(second-nearest),0,1);
    return gap**2*(.12+.88*variant**2);
}

function prepareScene(target,c,scene){
    const {width,height,background,depth,floorX,floorZ,floorGrain,coals,coalCells}=target;
    const [ox,oy,oz]=c.origin,[fx,fy,fz]=c.ahead,[rx,ry,rz]=c.right,[ux,uy,uz]=c.up;
    const f=height/(2*lens);
    const centerDepth=-ox*fx+(.13-oy)*fy-oz*fz;
    const bedX=width*.5+(-ox*rx+(.13-oy)*ry-oz*rz)*f/Math.max(.1,centerDepth);
    const bedY=height*.5-(-ox*ux+(.13-oy)*uy-oz*uz)*f/Math.max(.1,centerDepth);
    const extent=.54*f/Math.max(.12,centerDepth-.58);
    const woodScene=scene==='wood'||scene==='crosswind';
    const emberBed=scene==='embers';
    const swept=scene==='fireball'||scene==='groundburst';
    const nozzle=scene==='jet'||scene==='pulsing';
    const ray={ox,oy,oz,x:0,y:0,z:0},hit=[0,0,0];
    for(let y=0;y<height;y++){
        const v=(height*.5-y-.5)/f;
        for(let x=0;x<width;x++){
            const index=x+y*width,j=index*3,u=(x+.5-width*.5)/f;
            // Rays have unit view-axis depth. A ray parameter can be compared
            // with a projected voxel without normalizing its direction.
            ray.x=fx+u*rx+v*ux;ray.y=fy+u*ry+v*uy;ray.z=fz+u*rz+v*uz;
            let t=ray.y<-.000001?-oy/ray.y:1e4;
            floorGrain[index]=0;
            coals[index]=0;
            hit[0]=.0016;hit[1]=.0020;hit[2]=.0025;
            if(t>0&&t<600){
                const gx=ox+ray.x*t,gz=oz+ray.z*t,r2=gx*gx+gz*gz;
                const attenuation=.036/(1+4.8*r2*r2);
                const grain=.88+.12*Math.sin(gx*81+2*Math.sin(gz*37))*Math.cos(gz*97+gx*13);
                if(swept||emberBed||scene==='pulsing'){
                    const bounce=grain/(1+.49*r2);
                    hit[0]+=.014*bounce;hit[1]+=.011*bounce;hit[2]+=.010*bounce;
                    floorX[index]=gx;floorZ[index]=gz;floorGrain[index]=grain;
                }else {
                    hit[0]+=attenuation*grain;hit[1]+=attenuation*.31*grain;hit[2]+=attenuation*.09*grain;
                }
                if(emberBed){
                    const bed=gx*gx+(gz*1.3)**2;
                    const rough=.009*Math.sin(gx*43+5*gz)*Math.cos(gz*29-9*gx);
                    if(bed+rough<.20){
                        coals[index]=coalCrack(27*gx,27*gz)*Math.max(0,1-bed/.20);
                        const ix=clamp(Math.round((gx/domain+.5)*(nx-1)),1,nx-2);
                        const iz=clamp(Math.round((gz/domain+.5)*(nz-1)),1,nz-2);
                        coalCells[index]=ix+nx+iz*stride;
                        hit[0]=.006;hit[1]=.004;hit[2]=.003;
                    }
                }
                if(!woodScene&&!swept&&!emberBed&&Math.abs(x-bedX)<extent&&Math.abs(y-bedY)<extent){
                    const plane=(.052-oy)/ray.y,px=ox+plane*ray.x,pz=oz+plane*ray.z,r=Math.hypot(px,pz);
                    const edge=nozzle?.10:.29;
                    if(plane>0&&r<edge){
                        const rim=r>edge-.019;
                        hit[0]=rim?.083:.018;hit[1]=rim?.055:.017;hit[2]=rim?.04:.018;t=plane;
                    }
                }
            }
            if(woodScene&&Math.abs(x-bedX)<extent&&Math.abs(y-bedY)<extent){
                for(const log of logs)t=material(ray,t,log,hit);
            }
            depth[index]=t;background[j]=hit[0];background[j+1]=hit[1];background[j+2]=hit[2];
        }
    }
}

function makeTarget(context,width,height){
    const size=width*height,fineSize=fxCount*fyCount*fzCount;
    return {width,height,pixels:context.createImageData(width,height),depth:new Float32Array(size),
        background:new Float32Array(3*size),red:new Float32Array(size),green:new Float32Array(size),
        blue:new Float32Array(size),floorX:new Float32Array(size),floorZ:new Float32Array(size),
        floorGrain:new Float32Array(size),coals:new Float32Array(size),coalCells:new Uint16Array(size),
        kernels:footprintKernels(width),cameraKey:'',scene:'',revision:0,
        // Only occupied, visible samples build their projection entry. While
        // orbiting this costs one projection per hot sample, never a full-volume rebuild.
        stamp:new Uint32Array(fineSize),centers:new Int32Array(fineSize),codes:new Uint16Array(fineSize),
        sampleDepth:new Float32Array(fineSize),
        fineWorld:[fxCount,fyCount,fzCount].map((count,axis)=>Float32Array.from({length:count},
            (_,i)=>(i+.5)/count*(axis===1?ceiling:domain)-(axis===1?0:domain*.5)))};
}

function blend(target,kernel,cx,cy,depth,area,r,g,b,ext){
    const {width,height,red,green,blue,depth:surface}=target;
    const center=cy*width+cx,{weights,offsets,xs,ys}=kernel;
    const boundary=cx<8||cy<8||cx>=width-8||cy>=height-8;
    for(let k=0;k<offsets.length;k++){
        if(boundary&&(cx+xs[k]<0||cx+xs[k]>=width||cy+ys[k]<0||cy+ys[k]>=height))continue;
        const pixel=center+offsets[k];
        if(depth>surface[pixel]+.019)continue;
        const weight=weights[k]*area,trans=Math.max(.02,1-weight*ext);
        red[pixel]=red[pixel]*trans+weight*r;
        green[pixel]=green[pixel]*trans+weight*g;
        blue[pixel]=blue[pixel]*trans+weight*b;
    }
}

// Fine subcell material reconstruction and back-to-front optical integration.
// The active cells of the reacting volume produce the footprint primitives.
// The pressure solve, all transported fields and lighting evolve in three axes.
function raster(flow,c,target,opts,noise){
    const {width,height,red,green,blue,background,kernels,pixels}=target;
    const [wx,wy,wz]=target.fineWorld,[heat,fuel,soot,U,V,W]=flow.gas;
    const {ahead,right,up,origin}=c,focal=height/(2*lens);
    const [ox,oy,oz]=origin,[ax,ay,az]=ahead,[rx,ry,rz]=right,[ux,uy,uz]=up;
    const isBeauty=!opts.mode||opts.mode==='beauty',shared=opts.sharedLighting!==false;
    const finite=Boolean(flow.customSource?.spread),custom=Boolean(flow.customSource);
    const pulse=flow.preset==='pulsing', emberBed=flow.preset==='embers';
    let radiant=0;
    if(custom&&isBeauty&&shared) {
        let xCenter=0,yCenter=0,zCenter=0;
        for(let z=1;z<nz-1;z+=2)for(let y=1;y<ny-1;y+=2)for(let x=1;x<nx-1;x+=2){
            const i=x+y*nx+z*stride;
            const w=Math.max(0,heat[i]-.13)**1.7*(.2+.8*Math.min(.5,fuel[i]));
            radiant+=w;xCenter+=x*w;yCenter+=y*w;zCenter+=z*w;
        }
        const inv=1/Math.max(.00001,radiant);
        const lx=(xCenter*inv/(nx-1)-.5)*domain,lz=(zCenter*inv/(nz-1)-.5)*domain;
        const ly=yCenter*inv/(ny-1)*ceiling+.16,light=clamp(radiant*.013,0,.13);
        for(let i=0,j=0;i<red.length;i++,j+=3){
            const r2=(target.floorX[i]-lx)**2+(target.floorZ[i]-lz)**2+ly*ly;
            const falloff=target.floorGrain[i]*light*ly/((.11+r2)*Math.sqrt(.16+r2));
            red[i]=background[j]*.68+falloff;
            green[i]=background[j+1]*.68+falloff*.47;
            blue[i]=background[j+2]*.68+falloff*.11;
        }
    }else{
        if(shared&&isBeauty){
            for(let i=0;i<heat.length;i+=7)radiant+=heat[i]*Math.min(.5,fuel[i]);
        }
        const lamp=isBeauty?(shared?clamp(.30+radiant*.017,.30,1.77):.83):.16;
        for(let i=0,j=0;i<red.length;i++,j+=3){
            red[i]=background[j]*lamp;green[i]=background[j+1]*lamp;blue[i]=background[j+2]*lamp;
        }
    }
    if(emberBed&&isBeauty){
        for(let i=0;i<red.length;i++){
            const crack=target.coals[i];
            if(!crack)continue;
            const heatNear=heat[target.coalCells[i]];
            const ember=crack*(.085+2.1*heatNear);
            red[i]+=ember;green[i]+=.24*ember;blue[i]+=.026*ember;
        }
    }
    const [nX,nX2,nY,nY2,nZ,nZ2]=noise,t=flow.time;
    for(let i=0;i<nX.length;i++){nX[i]=Math.sin(i*.4+t*.3);nX2[i]=Math.cos(i*.88-t*.51);}
    for(let i=0;i<nY.length;i++){nY[i]=Math.cos(i*.155-t*1.95);nY2[i]=Math.sin(i*.23-t*2.6);}
    for(let i=0;i<nZ.length;i++){nZ[i]=Math.cos(i*.43+t*.56);nZ2[i]=Math.sin(i*.85-t*.73);}
    const footprint=opts.stableSampling===false?.88:1.20;
    const halfCell=domain/fxCount;
    // Sort the traversal along the dominant view axis; near/far never depend
    // on a camera orbit quadrant's texture-coordinate convention.
    const axis=Math.abs(ay)>Math.max(Math.abs(ax),Math.abs(az))?1:Math.abs(ax)>Math.abs(az)?0:2;
    const counts=[nx-1,ny-1,nz-1],indices=[0,0,0],d=[ax,ay,az];
    const order=[axis,(axis+1)%3,(axis+2)%3];
    for(let k=0;k<counts[order[0]];k++){
        indices[order[0]]=d[order[0]]<=0?k:counts[order[0]]-1-k;
        for(let m=0;m<counts[order[1]];m++){
            indices[order[1]]=d[order[1]]<=0?m:counts[order[1]]-1-m;
            for(let n=0;n<counts[order[2]];n++){
                indices[order[2]]=d[order[2]]<=0?n:counts[order[2]]-1-n;
                const x=indices[0],y=indices[1],z=indices[2],i=x+y*nx+z*stride;
                const vapor=custom?2.2*(soot[i]+soot[i+1]+soot[i+nx]+soot[i+stride]):soot[i];
                if(heat[i]+heat[i+1]+heat[i+nx]+heat[i+stride]+heat[i+stride+1]+heat[i+nx+1]+vapor<(emberBed?.057:.095) && (opts.mode!=='velocity'||Math.abs(U[i])+Math.abs(V[i])+Math.abs(W[i])<.08))continue;
                for(let dz=0;dz<2;dz++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
                    const xx=2*x+dx,yy=2*y+dy,zz=2*z+dz;
                    const fx=.25+dx*.5,fy=.25+dy*.5,fz=.25+dz*.5;
                    const h=interpolate(heat,i,fx,fy,fz),smoke=interpolate(soot,i,fx,fy,fz);
                    if(h+2*smoke<(emberBed?.021:.043)&&opts.mode!=='velocity')continue;
                    let r,g,b,ext;
                    if(isBeauty){
                        const gas=interpolate(fuel,i,fx,fy,fz);
                        const fold=.19*(nX[xx]*nY[yy]+nZ[zz]*nY2[yy]+.45*nX2[xx]*nZ2[zz]);
                        const front=gas+.22*h-.41+fold,den=1+1030*front*front;
                        const sheet=1/(den*den);
                        const temp=clamp((h-.105)*2.50+.18*nY2[yy]*nX[xx],0,1.28);
                        const fade=clamp((ceiling-wy[yy])/.30,0,1);
                        const emission=fade*temp*temp*(.012+1.44*sheet)*Math.min(1,h*4);
                        const scatter=smoke*fade*(shared?.0065:.0045);
                        r=emission*.92+scatter;g=emission*(.036+.25*temp*temp)+scatter*.90;
                        b=emission*(.001+.018*Math.max(0,temp-.45)**2)+scatter*.93;
                        ext=.023*fade*(smoke*3.6+temp*(.2+sheet));
                        if(custom){
                            const front=gas+.29*h-(finite?.57:.39)+fold*(finite?1.55:1);
                            const grazing=1/(1+310*front*front+2700*front**4);
                            const hot=clamp((h-(emberBed?.082:.11))*2.2,0,1.37);
                            const flame=(finite?.24:emberBed?1.21:.92)*hot**2.25*(.014+.986*grazing)*fade;
                            const incandescence=finite?clamp((h-.48)*1.5,0,1):0;
                            const blueBand=pulse?.26*Math.max(0,gas)*Math.max(0,h-.05)*Math.max(0,1-wy[yy]*3.8):0;
                            const haze=smoke*fade*(shared?(emberBed?.068:finite?.024:.025):.010);
                            r=flame*(1-.2*incandescence)+haze*.64+blueBand*.05;
                            g=flame*(.08+.42*hot+.24*incandescence)+haze*.68+blueBand*.40;
                            b=flame*(.01+.15*incandescence)+haze*.78+blueBand;
                            ext=fade*(.12*smoke+(finite?.038:.026)*hot*(.31+grazing));
                        }
                    }else if(opts.mode==='temperature'){
                        const v=Math.min(1,Math.max(0,h-.012));
                        r=v*v*.13;g=v*v*v*.078;b=v*(1-v)*.048;ext=.08*v;
                    }else if(opts.mode==='smoke'){
                        const v=Math.min(1,smoke*6);r=.11*v;g=.12*v;b=.132*v;ext=.12*v;
                    }else{
                        const ux=interpolate(U,i,fx,fy,fz),uy=interpolate(V,i,fx,fy,fz),uz=interpolate(W,i,fx,fy,fz);
                        const speed=Math.hypot(ux,uy,uz);
                        if(speed<.10)continue;
                        const level=clamp(speed*.65,0,1);
                        r=level*level*.16;g=level*.085;b=(1-level)*level*.15;ext=level*.08;
                    }
                    if(r+g+b<.00080)continue;
                    const cell=xx+fxCount*(yy+fyCount*zz);
                    if(target.stamp[cell]!==target.revision){
                        target.stamp[cell]=target.revision;
                        const px=wx[xx]-ox,py=wy[yy]-oy,pz=wz[zz]-oz;
                        const depth=px*ax+py*ay+pz*az;
                        target.sampleDepth[cell]=depth;target.codes[cell]=0;
                        if(depth<=.12)continue;
                        const scale=focal/depth;
                        const screenX=width*.5+(px*rx+py*ry+pz*rz)*scale;
                        const screenY=height*.5-(px*ux+py*uy+pz*uz)*scale;
                        if(screenX< -8||screenX>width+8||screenY< -8||screenY>height+8)continue;
                        const ix=Math.floor(screenX),iy=Math.floor(screenY);
                        const diameter=clamp(Math.round(2*footprint*halfCell*scale),2,14);
                        // Offset the signed coordinates so clipped border samples fit the encoding.
                        target.centers[cell]=((iy+8)<<16)|(ix+8);
                        target.codes[cell]=(diameter<<4)+(((screenY-iy)*4|0)<<2)+((screenX-ix)*4|0);
                    }
                    const code=target.codes[cell];
                    if(!code)continue;
                    const center=target.centers[cell],size=(code>>4)*.5/footprint;
                    blend(target,kernels[code],(center&65535)-8,(center>>16)-8,target.sampleDepth[cell],size*size,r,g,b,ext);
                }
            }
        }
    }
    const rgba=pixels.data,exposure=clamp(opts.exposure??1,.1,5);
    for(let i=0,j=0;i<red.length;i++,j+=4){
        rgba[j]=ambientLut(red[i]*exposure);rgba[j+1]=ambientLut(green[i]*exposure);
        rgba[j+2]=ambientLut(blue[i]*exposure);rgba[j+3]=255;
    }
}

export class CpuVolume {
    constructor(canvas,quality='low'){
        this.canvas=canvas;this.quality=quality;
        this.context=canvas.getContext('2d',{alpha:false});
        this.image=document.createElement('canvas');
        this.paint=this.image.getContext('2d',{alpha:false});
        this.flow=new CpuFlow();
        this.dims=this.flow.dims;
        this.noise=[fxCount,fxCount,fyCount,fyCount,fzCount,fzCount].map(n=>new Float32Array(n));
    }
    reset(params){this.flow.reset(params);}
    step(dt,params,time){this.flow.step(dt,params,time);}
    draw(opts={}){
        const w=this.canvas.width,h=this.canvas.height;
        if(w<1||h<1)return;
        const budget=pixelBudgets[opts.quality??this.quality]??pixelBudgets.low;
        const ratio=Math.min(1,Math.sqrt(budget/(w*h)));
        const width=Math.max(2,Math.round(w*ratio)),height=Math.max(2,Math.round(h*ratio));
        this.renderWidth=width;this.renderHeight=height;
        if(!this.target||width!==this.target.width||height!==this.target.height){
            this.image.width=width;this.image.height=height;this.target=makeTarget(this.paint,width,height);
        }
        const key=[opts.yaw??.38,opts.pitch??.14,opts.distance??3.9,opts.targetHeight??.98].join('/');
        const c=camera(opts),target=this.target;
        if(target.cameraKey!==key||target.scene!==this.flow.preset){
            prepareScene(target,c,this.flow.preset);target.cameraKey=key;target.scene=this.flow.preset;target.revision++;
        }
        const stable=opts.stableSampling!==false;
        if(target.stable!==stable){target.stable=stable;target.revision++;}
        raster(this.flow,c,target,opts,this.noise);
        this.paint.putImageData(target.pixels,0,0);
        this.context.imageSmoothingEnabled=true;
        this.context.drawImage(this.image,0,0,w,h);
    }
    brushAt(nxScreen,nyScreen,opts={}){
        const c=camera(opts),x=(nxScreen*2-1)*this.canvas.width/Math.max(1,this.canvas.height)*lens;
        const y=(1-2*nyScreen)*lens;
        const ray=c.ahead.map((v,i)=>v+x*c.right[i]+y*c.up[i]);
        const normalX=Math.sin(opts.yaw??.38),normalZ=Math.cos(opts.yaw??.38);
        const denom=normalX*ray[0]+normalZ*ray[2];
        const t=Math.abs(denom)<.00001?0:-(normalX*c.origin[0]+normalZ*c.origin[2])/denom;
        return [clamp(.5+(c.origin[0]+t*ray[0])/domain,.03,.97),clamp((c.origin[1]+t*ray[1])/ceiling,.035,.96),
            clamp(.5+(c.origin[2]+t*ray[2])/domain,.03,.97)];
    }
    destroy(){
        this.target=null;
        this.image.width=this.image.height=1;
    }
}

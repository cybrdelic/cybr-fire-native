import {lightFragment, lightHelper} from './lighting.js';

// The volume, opaque scene and glowing particles share camera and world coordinates.
// RGB16F scene radiance stays linear through the bloom pyramid and tone mapping.
const vertex = `#version 300 es
precision highp float;
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const common = `
uniform highp sampler2D uField, uVelocity;
uniform ivec3 uDims;
uniform int uTiles;
vec4 sampleGrid(highp sampler2D tex, vec3 world) {
    vec3 unit = (world + vec3(1.,0.,1.)) / vec3(2.,2.4,2.);
    vec3 p = clamp(unit * vec3(uDims) - .5, vec3(0.), vec3(uDims - 1));
    int z = int(p.z);
    int top = min(z + 1, uDims.z-1);
    vec2 size = vec2(textureSize(tex, 0));
    vec2 xy = p.xy + .5;
    vec2 a = xy + vec2(z % uTiles, z / uTiles) * vec2(uDims.xy);
    vec2 b = xy + vec2(top % uTiles, top / uTiles) * vec2(uDims.xy);
    return mix(textureLod(tex,a/size,0.), textureLod(tex,b/size,0.), fract(p.z));
}
`;

const irradianceFragment = `#version 300 es
precision highp float;
precision highp int;
` + common + lightFragment;

const camera = `
uniform vec3 uOrigin, uAhead, uRight, uUp;
uniform float uAspect;
const float lens = .32;
`;

const volume = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
in vec2 uv;
out vec4 frag;
uniform sampler3D uNoise;
uniform highp sampler2D uRestA, uRestB;
uniform int uHasRest;
uniform float uRestBlend;
uniform float uTime;
uniform int uPreset, uMode, uSteps;
` + camera + common + `
vec3 boxMin=vec3(-1.,0.,-1.), boxMax=vec3(1.,2.4,1.);
vec4 smoothNoise(vec3 p) {
    vec3 i=floor(p), f=fract(p);
    return textureLod(uNoise,(i+f*f*(3.-2.*f)+.5)/32.,0.);
}
float noise3(vec3 p) { return smoothNoise(p).r; }
float hash31(vec3 p) {
    p=fract(p*vec3(.1031,.11369,.13787));
    p+=dot(p,p.yzx+19.19);
    return fract((p.x+p.y)*p.z);
}
` + lightHelper + surfaceShader() + `
vec2 interval(vec3 ro, vec3 rd) {
    vec3 inv=sign(rd)/max(abs(rd),vec3(.000001));
    vec3 t0=(boxMin-ro)*inv, t1=(boxMax-ro)*inv;
    vec3 l=min(t0,t1), h=max(t0,t1);
    return vec2(max(0.,max(l.x,max(l.y,l.z))),min(h.x,min(h.y,h.z)));
}
vec3 temperaturePalette(float t) {
    vec3 c=mix(vec3(.021,.014,.15),vec3(.30,.038,.60),smoothstep(.03,.27,t));
    c=mix(c,vec3(1.,.065,.038),smoothstep(.25,.55,t));
    c=mix(c,vec3(1.,.7,.015),smoothstep(.52,.83,t));
    return mix(c,vec3(1.,1.,.89),smoothstep(.78,1.12,t));
}
// Frozen-noise breakup is advected upward on two scales; the resolved field
// determines where emission, soot and the thin subvoxel flame fronts can exist.
vec3 volumeLight(vec3 ro,vec3 rd,vec3 bg,float farT,vec3 emitter,float luminous) {
    vec2 hit=interval(ro,rd);
    hit.y=min(hit.y,farT);
    if(hit.y<=hit.x) return bg;
    float nominal=2./float(uSteps);
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint hash=pixel.x*0x8da6b343u ^ pixel.y*0xd8163841u;
    hash^=hash>>16u; hash*=0x7feb352du; hash^=hash>>15u;
    float jitter=.14+.72*float(hash&65535u)/65535.;
    float pos=hit.x;
    vec3 sum=vec3(0.);
    float tr=1.;
    // Spend fine samples on sheets and wider samples in optically smooth air.
    // The integration step uses world-space distance and a stable per-ray phase.
    for(int i=0;i<384;i++){
        if(pos>=hit.y || tr<.011) break;
        vec3 p=ro+rd*pos;
        vec4 base=sampleGrid(uField,p);
        float refinement=smoothstep(.035,.49,base.r);
        float ds=min(hit.y-pos,nominal*mix(2.05,.31,refinement));
        if(uMode!=0) ds=min(hit.y-pos,nominal*1.08);
        p=ro+rd*(pos+ds*(.16+.68*fract(jitter+float(i)*.61803399)));
        pos+=ds;
        if(base.r+base.g<.006 && uMode!=3) continue;
        float heat=base.r, soot=max(0.,base.g), fuel=max(0.,base.b);
        vec3 emission=vec3(0.), scatter=vec3(0.);
        float k=0.;
        if(uMode==0){
            // Rest positions travel with the resolved flow. Reinitialize two
            // coordinate maps on alternating phases and crossfade their noise.
            vec3 a=p-vec3(0.,uTime*.74,0.), b=a;
            if(uHasRest!=0){
                a=sampleGrid(uRestA,p).rgb*vec3(2.,2.4,2.)-vec3(1.,0.,1.);
                b=sampleGrid(uRestB,p).rgb*vec3(2.,2.4,2.)-vec3(1.,0.,1.);
            }
            vec3 freq=vec3(4.5,2.5,4.5), ph=vec3(3.7,8.9,1.9);
            vec3 warpA=smoothNoise(a*freq+ph).rgb-.5, warp=warpA;
            vec3 warpB=warpA;
            if(uHasRest!=0){
                warpB=smoothNoise(b*freq+ph).rgb-.5;
                warp=mix(warpA,warpB,uRestBlend);
            }
            vec3 wrinkled=p+warp*(.073+.10*smoothstep(.1,1.2,p.y));
            vec4 state=sampleGrid(uField,wrinkled);
            heat=state.r; soot=max(0.,state.g); fuel=max(0.,state.b);
            vec3 detailFreq=vec3(10.7,4.8,10.7);
            vec3 q=a*detailFreq+warpA*3.3;
            vec4 eddyA=smoothNoise(q);
            float fineA=noise3(q*2.14+1.2*eddyA.gbr);
            vec4 eddy=eddyA; float fine=fineA;
            if(uHasRest!=0){
                q=b*detailFreq+warpB*3.3;
                vec4 eddyB=smoothNoise(q);
                eddy=mix(eddyA,eddyB,uRestBlend);
                fine=mix(fineA,noise3(q*2.14+1.2*eddyB.gbr),uRestBlend);
            }
            float variation=(eddy.r-.5)*1.45+(fine-.5)*.48;
            // A filtered mixture-fraction isosurface emphasizes reaction sheets
            // where incoming fuel and oxidizer overlap. Cooler soot radiates less.
            float mixture=fuel/(.018+.24*max(0.,state.a)+fuel);
            float flamelet=.66*heat+.72*mixture-.70+.52*variation;
            float sheet=exp2(-91.*flamelet*flamelet);
            float hasHeat=smoothstep(.17,.50,heat+.13*variation);
            float richCore=1.-.73*smoothstep(.36,.68,mixture);
            float carbon=soot*clamp(.84+1.18*(eddy.b-.5)+.37*(fine-.5),.08,1.65);
            float gas=hasHeat*richCore*(.068+2.10*sheet);
            float flameT=clamp(.91*heat+.26*variation+.085*sheet,0.,1.07);
            float smokeK=carbon*5.4;
            float fireK=gas*(1.1+2.*carbon);
            k=smokeK+fireK;
            float intensity=exp2(8.65*(flameT-.69));
            vec3 glow=vec3(1.,.040+.295*pow(flameT,1.6),.0009+.029*pow(flameT,4.0));
            emission=glow*intensity*gas*(2.05+3.4*carbon);
            float hot=hasHeat;
            if(uPreset==2){
                float attachment=exp(-14.*max(0.,p.y-.075))*hot;
                emission+=vec3(.035,.30,1.5)*attachment*1.5;
            }
            if(smokeK>.009){
                vec3 lamp=emitter;
                vec3 lp=lamp-p;
                float lx=dot(lp,lp);
                float shadow=exp(-1.92*carbon);
                if(carbon>.055 && heat<.55 && uSteps>48){
                    vec3 shift=lp*min(1.,.25*inversesqrt(max(.002,lx)));
                    shadow*=exp(-.98*sampleGrid(uField,p+shift).g);
                }
                float forward=dot(normalize(lp+vec3(.0001)),rd)*.25+1.;
                scatter=smokeK*(vec3(.044,.052,.063)+luminous*vec3(.18,.052,.015)*shadow*forward/(lx+.24));
            }
        }else if(uMode==1){
            float denom=smoothstep(.018,.88,heat);
            k=denom*8.;
            emission=temperaturePalette(heat)*k*1.35;
        }else if(uMode==2){
            k=soot*14.;
            scatter=k*mix(vec3(.18,.21,.25),vec3(.93,.90,.86),exp(-1.85*soot));
        }else{
            vec3 v=sampleGrid(uVelocity,p).xyz;
            float speed=length(v);
            k=smoothstep(.18,1.65,speed)*(2.2+smoothstep(.04,.42,heat)*1.8);
            vec3 dir=v/max(speed,.001);
            vec3 c=mix(vec3(.012,.09,.30),vec3(.02,.55,.56),smoothstep(.23,1.04,speed));
            c=mix(c,vec3(.89,.97,.33),smoothstep(1.0,2.25,speed));
            emission=k*c*(.8+.2*abs(dir.y));
        }
        float opacity=1.-exp(-k*ds);
        sum+=tr*(emission+scatter)*(k>.0001?opacity/k:ds);
        tr*=1.-opacity;
    }
    return sum+tr*bg;
}
void main(){
    vec2 ndc=2.*uv-1.;
    vec3 rd=normalize(uAhead+lens*(ndc.x*uAspect*uRight+ndc.y*uUp));
    vec4 moment=vec4(0.);
    for(int i=0;i<4;i++){
        vec4 light=texelFetch(uLights,ivec2(i,0),0);
        moment+=vec4(light.xyz*light.w, light.w);
    }
    vec3 emitter=moment.xyz/max(moment.w,.00001);
    float farT;
    vec3 bg=surface(uOrigin,rd,farT);
    vec3 color=volumeLight(uOrigin,rd,bg,farT,emitter,min(2.,moment.w));
    frag=vec4(max(color,vec3(0.)),1.);
}`;

function surfaceShader(){
    return `// Clip a camera ray against one tapered, faceted piece of firewood.
// The bark coordinates run continuously around its axis, including when the camera orbits.
void sfLog(vec3 ro, vec3 rd, vec3 a, vec3 b, vec2 radii, float seed, inout float hitT, inout vec3 normal, inout vec3 local, inout vec3 woodInfo) {
    vec3 axis = b-a;
    float len = length(axis);
    axis /= len;
    vec3 side = normalize(cross(axis,vec3(0,1,0)));
    vec3 up = cross(side,axis);
    vec3 v = ro-a;
    vec3 p = vec3(dot(v,side)/radii.x,dot(v,up)/radii.y,dot(v,axis));
    vec3 d = vec3(dot(rd,side)/radii.x,dot(rd,up)/radii.y,dot(rd,axis));
    float nearT = -1000., farT = hitT;
    vec3 hitN = vec3(0);
    // End grain is cut on a slight bias.
    float endTilt = .016*sin(seed);
    float planeZ = p.z+endTilt*p.x;
    float dirZ = d.z+endTilt*d.x;
    if (abs(dirZ)<.00001) {
        if (planeZ<0. || planeZ>len) return;
    } else {
        float z0 = -planeZ/dirZ, z1 = (len-planeZ)/dirZ;
        nearT = min(z0,z1); farT = min(farT,max(z0,z1));
        hitN = (dirZ>0. ? -1. : 1.)*normalize(axis+endTilt/radii.x*side);
        if (nearT>farT) return;
    }
    float face = 1.;
    // Circle bound rejects background and other logs before testing the bevel planes.
    float proj = clamp(-dot(p.xy,d.xy)/max(.00001,dot(d.xy,d.xy)), max(0.,nearT),max(0.,farT));
    if (length(p.xy+d.xy*proj)>1.2) return;
    for (int j=0;j<8;j++) {
        float theta = float(j)*.7853982+seed*.71;
        vec2 n = vec2(cos(theta),sin(theta));
        float edge = .89+.065*sin(float(j)*3.17+seed*5.);
        if(j==0) edge -= .26;
        float taper=.09/len;
        float dist = edge-dot(n,p.xy)-p.z*taper;
        float speed=dot(n,d.xy)+d.z*taper;
        if(abs(speed)<.00001) {
            if(dist < 0.) return;
        } else {
            float t=dist/speed;
            if(speed<0. && t>nearT) {
                nearT=t;
                hitN=normalize(side*n.x/radii.x+up*n.y/radii.y+axis*taper);
                face=0.;
            }
            if(speed>0.) farT=min(farT,t);
            if(nearT>farT) return;
        }
    }
    if(nearT<.001 || nearT>=hitT) return;
    hitT=nearT; normal=hitN; local=p+nearT*d;
    woodInfo=vec3(seed,face,len);
}

// Solid capped cylinder with its upper face visible across a shallow burner pan.
void sfCylinder(vec3 ro,vec3 rd,float radius,float lowY,float highY,inout float hitT,inout vec3 norm,inout float part,float kind) {
    float aa=dot(rd.xz,rd.xz), bb=dot(ro.xz,rd.xz);
    float h=bb*bb-aa*(dot(ro.xz,ro.xz)-radius*radius);
    if(h>=0. && aa>.00001) {
        float t=(-bb-sqrt(h))/aa;
        vec3 p=ro+rd*t;
        if(t>.001 && t<hitT && p.y>=lowY && p.y<=highY) {
            hitT=t; norm=normalize(vec3(p.x,0.,p.z)); part=kind;
        }
    }
    if(abs(rd.y)>.00001) {
        float t=(highY-ro.y)/rd.y;
        vec3 p=ro+rd*t;
        if(t>.001 && t<hitT && dot(p.xz,p.xz)<radius*radius) {
            hitT=t; norm=vec3(0,1,0); part=kind+1.;
        }
    }
}

vec3 surface(vec3 ro,vec3 rd,out float hitT) {
    hitT=1000.;
    vec3 ambient=vec3(.010,.013,.016);
    bool wood=(uPreset==0 || uPreset==3);
    if(rd.y<-.00001) hitT=min(1000.,-ro.y/rd.y);
    vec3 local=vec3(0),norm=vec3(0,1,0),info=vec3(-1);
    float part=-1.;
    // Bound the complete fire bed before doing any detailed shape tests.
    vec3 off=ro-vec3(0,.11,0);
    float proj=max(-dot(off,rd),0.);
    if(dot(off+proj*rd,off+proj*rd)<.40) {
        if(wood) {
            sfLog(ro,rd,vec3(-.39,.070,-.11),vec3(.30,.066,-.17),vec2(.080,.068),1.2,hitT,norm,local,info);
            sfLog(ro,rd,vec3(-.32,.068,.18),vec3(.39,.070,.085),vec2(.075,.067),2.7,hitT,norm,local,info);
            sfLog(ro,rd,vec3(-.28,.076,-.31),vec3(.17,.087,.30),vec2(.079,.064),3.4,hitT,norm,local,info);
            sfLog(ro,rd,vec3(-.20,.127,.30),vec3(.19,.176,-.32),vec2(.082,.068),4.1,hitT,norm,local,info);
            sfLog(ro,rd,vec3(.38,.065,.24),vec3(-.22,.202,-.19),vec2(.052,.048),5.2,hitT,norm,local,info);
            sfLog(ro,rd,vec3(-.37,.045,.10),vec3(.24,.185,-.08),vec2(.042,.040),6.9,hitT,norm,local,info);
        } else {
            float r=(uPreset==2 ? .138 : .405);
            sfCylinder(ro,rd,r*.91,.002,.052,hitT,norm,part,1.);
            sfCylinder(ro,rd,r,.050,.074,hitT,norm,part,3.);
        }
    }
    if(hitT>999.) return ambient*.60;
    vec3 p = ro+rd*hitT;
    if (length(p.xz)>12.) return ambient*.60;
    SfIrradiance incident=sfIncident(p,norm,-rd,wood,info.x);
    float hotBed=smoothstep(.001,.39,incident.flux);
    vec3 cool=vec3(.10,.132,.16)*(.48+.52*max(0.,dot(norm,normalize(vec3(-.6,.8,-.4)))));

    if(info.x>0.) {
        float seed=info.x;
        float angle=atan(local.y,local.x)/6.2831853;
        float s=local.z;
        float ringDist=length(local.xy+vec2(.08*sin(seed),.10*cos(seed)));
        vec3 noisePoint=vec3(local.xy*2.,s*7.+seed*44.);
        float broad=noise3(noisePoint);
        float mid=noise3(noisePoint*3.1);
        float fine=noise3(noisePoint*9.2);
        float wave=angle*28.+broad*1.45+.35*mid;
        float longCrack=abs(fract(wave)-.5);
        float crossCrack=abs(fract(s*17.+angle*.35+mid*.67+broad)-.5);
        float seam=min(longCrack,1.68*crossCrack);
        float warmth=clamp(1.1-length(p.xz)*2.1+.48*(broad-.5),.09,1.);
        float split=1.-smoothstep(.018,.105,seam);
        float scale=.35+.65*smoothstep(.04,.25,seam);
        vec3 albedo=mix(vec3(.036,.025,.022),vec3(.16,.147,.128),smoothstep(.36,.81,broad*.55+mid*.32+fine*.13))*scale;
        float embers=split*pow(warmth,2.2)*( .20+.83*mid);
        float ash=smoothstep(.63,.83,mid*.48+broad*.34+fine*.18);
        albedo+=vec3(.18,.165,.143)*ash*scale;
        if(info.y>.5) {
            float rings=abs(fract(ringDist*13.+broad*.64)-.5);
            float radial=abs(fract(angle*9.+broad*.75+seed)-.5);
            float burl=.42+.58*smoothstep(.07,.28,rings);
            albedo=mix(vec3(.047,.034,.023),vec3(.24,.115,.044),broad*.7)*burl;
            embers=warmth*.62*(1.-smoothstep(.016,.085,radial))*smoothstep(.14,.36,ringDist);
            albedo*=.5+.5*smoothstep(.015,.085,radial);
        }
        vec3 coal=vec3(1.98,.16,.011)*hotBed*embers*(uMode==0 ? 1. : .15);
        vec3 shade=albedo*(cool+.08*incident.fill+.93*incident.direct);
        return shade+coal;
    }
    if (part>0.) {
        float r=(uPreset==2 ? .138 : .405);
        float rr=length(p.xz)/r;
        float grain=noise3(p*193.);
        bool rim=rr>.927 || part==3.;
        vec3 metal=mix(vec3(.045,.042,.039),vec3(.115,.100,.083),grain);
        vec3 steel=metal*(cool+.047*incident.fill+.51*incident.direct)+.31*incident.specular;
        if(rim) return steel+vec3(.036,.023,.015)*(1.-smoothstep(.010,.045,abs(rr-.966)));
        if(uPreset==2) {
            vec2 cell=fract(p.xz*59.+.5)-.5;
            float open=1.-smoothstep(.21,.31,length(cell));
            return mix(steel*.70,vec3(.018,.006,.003),open);
        }
        return vec3(.011,.009,.008)+.096*incident.gloss*(.56+.44*grain);
    }
    float r=length(p.xz);
    if(r>12.) return ambient*.60;
    float broad=noise3(vec3(p.xz*5.1,2.));
    float mid=noise3(vec3(p.xz*29.,8.));
    float grains=noise3(vec3(p.xz*145.,18.));
    float detail=.34*broad+.31*mid+.35*grains;
    vec3 earth=mix(vec3(.043,.043,.045),vec3(.122,.115,.105),detail);
    float soot=wood ? 1.-smoothstep(.27,.77,r+.12*(broad-.5)) : 0.;
    earth*=1.-.48*soot;
    float ash=wood ? (1.-smoothstep(.30,.64,r+.12*broad))*smoothstep(.43,.83,mid*.57+grains*.43) : 0.;
    earth+=vec3(.095,.088,.075)*ash;
    float contact=wood ? mix(.33,1.,smoothstep(.12,.46,r)) : mix(.26,1.,smoothstep(.28,.70,r));
    vec3 ground=earth*(cool*.74+incident.direct*mix(contact,1.,wood ? .68 : 0.)*(wood ? .83 : .62));
    if(wood && r<.45 && uMode==0) {
        float bed=(1.-smoothstep(.19,.45,r))*smoothstep(.54,.77,broad*.32+mid*.45+grains*.23);
        ground+=vec3(.95,.091,.004)*bed*hotBed;
    }
    return mix(ground,ambient*.60,smoothstep(2.,11.,r));
}
`;
}

const bloomFragment = `#version 300 es
precision highp float;
in vec2 uv; out vec4 frag;
uniform sampler2D uImage;
uniform bool uThreshold;
void main() {
    vec2 d=1.15/vec2(textureSize(uImage,0));
    vec3 c=texture(uImage,uv).rgb*4.;
    c+=texture(uImage,uv+d).rgb+texture(uImage,uv-d).rgb;
    c+=texture(uImage,uv+d*vec2(-1,1)).rgb+texture(uImage,uv+d*vec2(1,-1)).rgb;
    c+=2.*(texture(uImage,uv+vec2(d.x,0)).rgb+texture(uImage,uv-vec2(d.x,0)).rgb+
           texture(uImage,uv+vec2(0,d.y)).rgb+texture(uImage,uv-vec2(0,d.y)).rgb);
    c*=.0625;
    if(uThreshold){float peak=max(max(c.r,c.g),c.b);c*=smoothstep(.52,1.6,peak)*min(1.,3.5/max(peak,.001));}
    frag=vec4(c,1.);
}`;

const outputFragment = `#version 300 es
precision highp float;
in vec2 uv; out vec4 frag;
uniform sampler2D uScene,uNear,uMid,uWide;
uniform float uExposure;
uniform int uMode;
void main(){
    vec3 c=texture(uScene,uv).rgb;
    if(uMode==0){
        // A one-pixel edge-aware reconstruction reduces quadrature noise while
        // preserving the contrast of silhouettes and thin bright filaments.
        vec2 px=1./vec2(textureSize(uScene,0));
        vec3 sum=c*1.7; float weight=1.7;
        const vec2 crossTap[4]=vec2[4](vec2(-1,0),vec2(1,0),vec2(0,-1),vec2(0,1));
        float guide=log2(1.+max(c.r,max(c.g,c.b)));
        for(int k=0;k<4;k++){
            vec3 n=texture(uScene,uv+px*crossTap[k]).rgb;
            float delta=log2(1.+max(n.r,max(n.g,n.b)))-guide;
            float w=exp2(-8.*delta*delta);
            sum+=w*n;weight+=w;
        }
        c=mix(c,sum/weight,.82*smoothstep(.015,.25,guide));
        vec2 d=.84/vec2(textureSize(uWide,0));
        vec3 broad=texture(uWide,uv).rgb*4.;
        broad+=texture(uWide,uv+d).rgb+texture(uWide,uv-d).rgb;
        broad+=texture(uWide,uv+d*vec2(-1,1)).rgb+texture(uWide,uv+d*vec2(1,-1)).rgb;
        c+=.065*texture(uNear,uv).rgb+.047*texture(uMid,uv).rgb+.033*broad/8.;
    }
    c*=exp2(uExposure);
    c=clamp((c*(2.51*c+.03))/(c*(2.43*c+.59)+.14),0.,1.);
    c=mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(vec3(.0031308),c));
    float grain=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))))-.5;
    frag=vec4(c+grain/255.,1.);
}`;

const sparksUpdate = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec4 aState;
out vec4 nextState;
uniform float uDt,uTime;
uniform int uPreset;
` + common + `
float random1(float p){return fract(sin(p*41.337+11.23)*43758.545);}
void main(){
    float id=float(gl_VertexID);
    vec4 s=aState;
    float life=.68+1.60*random1(id+4.);
    float age=s.w+uDt;
    if(age>=life){
        float a=random1(id+floor(uTime*2.))*6.283185;
        float r=.30*sqrt(random1(19.+id+floor(uTime*1.3)));
        s.xyz=vec3(cos(a)*r,.17+.08*random1(id+3.),sin(a)*r);
        float temp=sampleGrid(uField,s.xyz).r;
        age=-.25-random1(3.+id+floor(uTime))*(temp>.14?2.7:5.5);
        if(temp<.1) s.y=-1.;
    }
    if(s.w<0. && age>=0. && sampleGrid(uField,s.xyz).r<.14) s.y=-1.;
    if(age>=0. && s.y>=0.){
        vec3 flow=sampleGrid(uVelocity,s.xyz).xyz;
        vec2 drift=vec2(sin(uTime*3.6+id*3.5+s.y*5.),cos(uTime*2.9+id+s.y*7.))*.12;
        s.xyz+=uDt*(flow*(.44+.28*random1(id+91.))+vec3(drift.x,.24-.13*age,drift.y));
    }
    nextState=vec4(s.xyz,age);
    gl_Position=vec4(0.);
}`;

const sparksVertex = `#version 300 es
precision highp float;
layout(location=0) in vec4 aState;
uniform float uHeight,uTime;
out float brightness;
` + camera + `
void main(){
    float id=float(gl_VertexID);
    float seed=fract(sin(id*41.337+11.23)*43758.545);
    vec3 d=aState.xyz-uOrigin;
    float z=dot(d,uAhead);
    gl_Position=vec4(dot(d,uRight)/(lens*uAspect),dot(d,uUp)/lens,0.,z);
    float life=.68+1.60*fract(sin((id+4.)*41.337+11.23)*43758.545);
    brightness=smoothstep(0.,.11,aState.w)*(1.-smoothstep(.40*life,life,aState.w));
    if(aState.y<.13 || aState.y>2.36) brightness=0.;
    brightness*=.7+.3*sin(uTime*7.1+id);
    gl_PointSize=clamp(uHeight*(.0035+.005*seed)/max(.13,z),1.4,9.);
}`;

const sparksFragment = `#version 300 es
precision highp float;
in float brightness;
out vec4 frag;
void main() {
    vec2 p=gl_PointCoord*2.-1.;
    float strength=brightness*exp(-dot(p,p)*4.1);
    frag=vec4(vec3(5.8,1.17,.082)*strength,0.);
}`;

const skipColor = `#version 300 es
precision highp float;
out vec4 color;
void main(){color=vec4(0.);}
`;

function buildProgram(gl, vs, fs, feedback=false) {
    const program=gl.createProgram();
    const shaders=[];
    try {
        for(const [kind,source] of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]) {
            const shader=gl.createShader(kind);
            shaders.push(shader);
            gl.shaderSource(shader,source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
            gl.attachShader(program,shader);
        }
        if (feedback) gl.transformFeedbackVaryings(program,['nextState'],gl.INTERLEAVED_ATTRIBS);
        gl.linkProgram(program);
        if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
        const result={program};
        for(let i=0;i<gl.getProgramParameter(program,gl.ACTIVE_UNIFORMS);i++){
            const name=gl.getActiveUniform(program,i).name;
            result[name]=gl.getUniformLocation(program,name);
        }
        return result;
    }catch(e){
        gl.deleteProgram(program);
        throw e;
    }finally{
        for(const shader of shaders)gl.deleteShader(shader);
    }
}

function basis({yaw=.38,pitch=.14,distance=3.9}={}){
    const sy=Math.sin(yaw),cy=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch);
    const ahead=[-sy*cp,-sp,-cy*cp], right=[cy,0,-sy],up=[-sy*sp,cp,-cy*sp];
    const origin=[-ahead[0]*distance,.98-ahead[1]*distance,-ahead[2]*distance];
    return {ahead,right,up,origin};
}

export function cameraRay(ndcX,ndcY,aspect,opts={}){
    const {origin,ahead,right,up}=basis(opts);
    const v=ahead.map((x,i)=>x+.32*(ndcX*aspect*right[i]+ndcY*up[i]));
    const n=Math.hypot(...v);
    return {origin,direction:v.map(x=>x/n)};
}

function noiseVolume(gl) {
    const n=32, data=new Uint8Array(n*n*n*4);
    let x=0x8e626ff1;
    for(let i=0;i<data.length;i++){
        x^=x<<13;x^=x>>>17;x^=x<<5;
        data[i]=x>>>24;
    }
    const tex=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D,tex);
    gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA8,n,n,n,0,gl.RGBA,gl.UNSIGNED_BYTE,data);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_3D,null);
    return tex;
}

export class Renderer{
    constructor(gl){
        this.gl=gl;
        this.programs={};
        this.targets=[];
        this.sparkBuffers=[];
        this.sparkVaos=[];
        this.numSparks=100;
        this.sparkIndex=0;
        this.lastTime=null;
        try{
            gl.getExtension('EXT_color_buffer_float');
            const p=this.programs;
            p.irradiance=buildProgram(gl,vertex,irradianceFragment);
            p.volume=buildProgram(gl,vertex,volume);
            p.bloom=buildProgram(gl,vertex,bloomFragment);
            p.composite=buildProgram(gl,vertex,outputFragment);
            p.update=buildProgram(gl,sparksUpdate,skipColor,true);
            p.sparks=buildProgram(gl,sparksVertex,sparksFragment);
            this.noise=noiseVolume(gl);
            this.lights=this.target(4,1);
            this.linear=gl.createSampler();
            gl.samplerParameteri(this.linear,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
            gl.samplerParameteri(this.linear,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
            gl.samplerParameteri(this.linear,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
            gl.samplerParameteri(this.linear,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
            this.vao=gl.createVertexArray();
            this.feedback=gl.createTransformFeedback();
            for(let i=0;i<2;i++){
                const buf=gl.createBuffer(), vao=gl.createVertexArray();
                this.sparkBuffers.push(buf);
                this.sparkVaos.push(vao);
                gl.bindVertexArray(vao);
                gl.bindBuffer(gl.ARRAY_BUFFER,buf);
                gl.bufferData(gl.ARRAY_BUFFER,this.numSparks*16,gl.DYNAMIC_COPY);
                gl.enableVertexAttribArray(0);
                gl.vertexAttribPointer(0,4,gl.FLOAT,false,16,0);
            }
            gl.bindVertexArray(null);
            gl.bindBuffer(gl.ARRAY_BUFFER,null);
        }catch(e){
            this.destroy();
            throw e;
        }
    }

    target(w,h){
        const gl=this.gl, tex=gl.createTexture(),fb=gl.createFramebuffer();
        gl.bindTexture(gl.TEXTURE_2D,tex);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,w,h,0,gl.RGBA,gl.HALF_FLOAT,null);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
        if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE){
            gl.deleteTexture(tex);gl.deleteFramebuffer(fb);
            throw Error('Half-float render targets are unavailable.');
        }
        return {w,h,fb,tex};
    }
    clearTargets(){
        const gl=this.gl;
        for(const t of this.targets){gl.deleteTexture(t.tex);gl.deleteFramebuffer(t.fb);}
        this.targets=[];
    }
    bindTarget(target){
        const gl=this.gl;
        gl.bindFramebuffer(gl.FRAMEBUFFER,target.fb);
        gl.viewport(0,0,target.w,target.h);
    }
    texture(unit,target,texture,location,sampler=null){
        const gl=this.gl;
        gl.activeTexture(gl.TEXTURE0+unit);
        gl.bindTexture(target,texture);
        gl.bindSampler(unit,sampler);
        gl.uniform1i(location,unit);
    }
    grid(program,fluid,velocity=true){
        const gl=this.gl;
        this.texture(0,gl.TEXTURE_2D,fluid.field,program.uField,this.linear);
        if(velocity)this.texture(1,gl.TEXTURE_2D,fluid.velocity,program.uVelocity,this.linear);
        gl.uniform3iv(program.uDims,fluid.dims);
        gl.uniform1i(program.uTiles,fluid.tiles);
    }
    camera(program,b,aspect){
        const gl=this.gl;
        gl.uniform3fv(program.uOrigin,b.origin);
        gl.uniform3fv(program.uAhead,b.ahead);
        gl.uniform3fv(program.uRight,b.right);
        gl.uniform3fv(program.uUp,b.up);
        gl.uniform1f(program.uAspect,aspect);
    }
    resetSparks(){
        const gl=this.gl, initial=new Float32Array(this.numSparks*4);
        for(let i=0;i<this.numSparks;i++){
            const a=i*2.39996323, r=.30*Math.sqrt((i+.5)/this.numSparks);
            initial[i*4]=Math.cos(a)*r;
            initial[i*4+1]=.16+.08*Math.sin(i*11.)**2;
            initial[i*4+2]=Math.sin(a)*r;
            initial[i*4+3]=-3.4*((i*73%101)/101);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER,this.sparkBuffers[0]);
        gl.bufferSubData(gl.ARRAY_BUFFER,0,initial);
        gl.bindBuffer(gl.ARRAY_BUFFER,null);
        this.sparkIndex=0;
    }
    sparks(fluid,b,aspect,time,preset,visible){
        const gl=this.gl, p=this.programs;
        const delta=time-(this.lastTime??time);
        if(this.lastTime===null || delta<0 || this.lastPreset!==preset)this.resetSparks();
        this.lastTime=time;this.lastPreset=preset;
        if(preset!==0 && preset!==3)return;
        if(delta>0){
            const u=p.update,dest=1-this.sparkIndex;
            gl.useProgram(u.program);
            this.grid(u,fluid);
            gl.uniform1f(u.uTime,time);
            gl.uniform1f(u.uDt,Math.min(.12,delta));
            gl.bindVertexArray(this.sparkVaos[this.sparkIndex]);
            gl.enable(gl.RASTERIZER_DISCARD);
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,this.feedback);
            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,this.sparkBuffers[dest]);
            gl.beginTransformFeedback(gl.POINTS);
            gl.drawArrays(gl.POINTS,0,this.numSparks);
            gl.endTransformFeedback();
            gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER,0,null);
            gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK,null);
            gl.disable(gl.RASTERIZER_DISCARD);
            this.sparkIndex=dest;
        }
        if(!visible)return;
        const u=p.sparks;
        gl.useProgram(u.program);
        this.camera(u,b,aspect);
        gl.uniform1f(u.uHeight,gl.drawingBufferHeight);
        gl.uniform1f(u.uTime,time);
        gl.bindVertexArray(this.sparkVaos[this.sparkIndex]);
        // Explicit floating-point blending support varies by implementation.
        gl.enable(gl.BLEND);
        gl.blendEquation(gl.FUNC_ADD);
        gl.blendFunc(gl.ONE,gl.ONE);
        gl.drawArrays(gl.POINTS,0,this.numSparks);
        gl.disable(gl.BLEND);
    }
    draw(fluid,options={}){
        const gl=this.gl, w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
        if(!w || !h || gl.isContextLost())return;
        const aspect=w/h, b=basis(options);
        const mode=({beauty:0,temperature:1,smoke:2,velocity:3})[options.mode]??0;
        const preset=({wood:0,pool:1,jet:2,crosswind:3})[options.preset]??0;
        const time=options.time??0;
        gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);
        gl.disable(gl.SCISSOR_TEST);gl.disable(gl.RASTERIZER_DISCARD);
        gl.colorMask(true,true,true,true);
        if(!this.targets.length || this.targets[0].w!==w || this.targets[0].h!==h){
            this.clearTargets();
            for(const div of [1,3,8,18])this.targets.push(this.target(Math.max(1,Math.floor(w/div)),Math.max(1,Math.floor(h/div))));
        }
        const [scene,near,mid,wide]=this.targets;
        const p=this.programs;
        gl.bindVertexArray(this.vao);
        this.bindTarget(this.lights);
        gl.useProgram(p.irradiance.program);
        this.grid(p.irradiance,fluid,false);
        gl.drawArrays(gl.TRIANGLES,0,3);
        this.bindTarget(scene);
        gl.useProgram(p.volume.program);
        this.camera(p.volume,b,aspect);
        this.grid(p.volume,fluid);
        this.texture(2,gl.TEXTURE_3D,this.noise,p.volume.uNoise);
        this.texture(6,gl.TEXTURE_2D,this.lights.tex,p.volume.uLights,this.linear);
        // WebGL validates every active sampler even if its branch is disabled.
        const rest=Boolean(fluid.restA && fluid.restB);
        this.texture(4,gl.TEXTURE_2D,rest?fluid.restA:fluid.field,p.volume.uRestA,this.linear);
        this.texture(5,gl.TEXTURE_2D,rest?fluid.restB:fluid.field,p.volume.uRestB,this.linear);
        gl.uniform1i(p.volume.uHasRest,rest?1:0);
        gl.uniform1f(p.volume.uRestBlend,fluid.restBlend??0);
        gl.uniform1i(p.volume.uSteps,({low:46,medium:72,high:96,ultra:112})[options.quality]??64);
        gl.uniform1f(p.volume.uTime,time);
        gl.uniform1i(p.volume.uMode,mode);
        gl.uniform1i(p.volume.uPreset,preset);
        gl.drawArrays(gl.TRIANGLES,0,3);
        this.sparks(fluid,b,aspect,time,preset,mode===0);
        gl.bindVertexArray(this.vao);
        if(mode===0){
            gl.useProgram(p.bloom.program);
            for(let i=1;i<this.targets.length;i++){
                this.bindTarget(this.targets[i]);
                this.texture(0,gl.TEXTURE_2D,this.targets[i-1].tex,p.bloom.uImage);
                gl.uniform1i(p.bloom.uThreshold,i===1?1:0);
                gl.drawArrays(gl.TRIANGLES,0,3);
            }
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER,null);
        gl.viewport(0,0,w,h);
        gl.useProgram(p.composite.program);
        this.texture(0,gl.TEXTURE_2D,scene.tex,p.composite.uScene);
        this.texture(1,gl.TEXTURE_2D,near.tex,p.composite.uNear);
        this.texture(2,gl.TEXTURE_2D,mid.tex,p.composite.uMid);
        this.texture(3,gl.TEXTURE_2D,wide.tex,p.composite.uWide);
        gl.uniform1i(p.composite.uMode,mode);
        gl.uniform1f(p.composite.uExposure,options.exposure??0);
        gl.drawArrays(gl.TRIANGLES,0,3);
        for(let i=0;i<7;i++)gl.bindSampler(i,null);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindVertexArray(null);
    }
    destroy(){
        const gl=this.gl;
        this.clearTargets();
        for(const p of Object.values(this.programs))gl.deleteProgram(p.program);
        for(const b of this.sparkBuffers)gl.deleteBuffer(b);
        for(const v of this.sparkVaos)gl.deleteVertexArray(v);
        if(this.lights){gl.deleteFramebuffer(this.lights.fb);gl.deleteTexture(this.lights.tex);}
        if(this.noise)gl.deleteTexture(this.noise);
        if(this.linear)gl.deleteSampler(this.linear);
        if(this.vao)gl.deleteVertexArray(this.vao);
        if(this.feedback)gl.deleteTransformFeedback(this.feedback);
    }
}

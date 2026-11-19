import {lightFragment, lightHelper} from './lighting.js';
import {presetCodes} from './presets.js';

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

const gridSampling = `
const vec3 gridScale=vec3(2.,2.4,2.);
const vec3 gridOffset=vec3(1.,0.,1.);
vec4 sampleAtlas(highp sampler2D tex, vec3 p, ivec3 dims, int tiles) {
    p=clamp(p,vec3(0.),vec3(dims-1));
    int z=int(p.z);
    int top=min(z+1,dims.z-1);
    vec2 xy=p.xy+.5;
    vec2 a=xy+vec2(z%tiles,z/tiles)*vec2(dims.xy);
    vec2 b=xy+vec2(top%tiles,top/tiles)*vec2(dims.xy);
    vec2 rec=1./vec2(textureSize(tex,0));
    return mix(textureLod(tex,a*rec,0.),textureLod(tex,b*rec,0.),fract(p.z));
}
`;

const common = gridSampling + `
uniform highp sampler2D uField, uVelocity;
uniform ivec3 uDims;
uniform int uTiles;
vec4 sampleGrid(highp sampler2D tex, vec3 world) {
    vec3 p=(world+gridOffset)/gridScale*vec3(uDims)-.5;
    return sampleAtlas(tex,p,uDims,uTiles);
}
`;

const lightingHeader = `#version 300 es
precision highp float;
precision highp int;
` + common;
const irradianceFragment = lightingHeader + lightFragment;

// Light transport through the smoke lives on a lower resolution grid. Every
// camera ray reads the same transmittance and therefore agrees when it orbits.
const smokeLightingFragment = `#version 300 es
precision highp float;
precision highp int;
out vec4 frag;
uniform highp sampler2D uLights;
uniform ivec3 uLightDims;
uniform int uLightTiles,uLightProbes;
uniform bool uLightShading;
` + common + `
float sootExtinction(vec3 p) {
    float soot=max(0.,sampleGrid(uField,p).g);
    float carbon=soot*.86;
    return 4.8*carbon*(.46+.54*sqrt(carbon+.02));
}
// Include all texels that could contribute to trilinear sampling in the cell.
float occupiedCell(ivec3 cell) {
    ivec3 start=cell*4;
    ivec3 stop=min(uDims-1,start+4);
    start=max(ivec3(0),start-1);
    for(int z=start.z;z<=stop.z;z++){
        ivec2 tile=ivec2(z%uTiles,z/uTiles)*uDims.xy;
        for(int y=start.y;y<=stop.y;y++){
            for(int x=start.x;x<=stop.x;x++){
                vec2 state=texelFetch(uField,tile+ivec2(x,y),0).rg;
                if(state.r+state.g>=.0045)return 1.;
            }
        }
    }
    return 0.;
}
void main() {
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    ivec3 cell=ivec3(pixel%uLightDims.xy,pixel.x/uLightDims.x+(pixel.y/uLightDims.y)*uLightTiles);
    if(cell.z>=uLightDims.z){frag=vec4(0);return;}
    vec3 pos=(vec3(cell)+.5)/vec3(uLightDims)*gridScale-gridOffset;
    float occupied=occupiedCell(cell);
    if(!uLightShading || occupied<.5){frag=vec4(0.,1.,occupied,1.);return;}
    // The solver has an open top. Integrate the zenith optical depth to it.
    float zenith=0.;
    float height=2.4-pos.y;
    for(int j=0;j<4;j++) zenith+=sootExtinction(pos+vec3(0.,height*(float(j)+.5)*.25,0.));
    float sky=exp(-.25*height*zenith);
    float radiant=0.;
    if(uLightProbes==1) {
        // Pool distant subgrid emitters on low quality.
        vec4 moment=vec4(0.);
        for(int k=0;k<4;k++){
            vec4 light=texelFetch(uLights,ivec2(k,0),0);
            moment+=vec4(light.xyz*light.w,light.w);
        }
        if(moment.w>.0001) {
            vec3 ray=moment.xyz/moment.w-pos;
            float d2=dot(ray,ray);
            float tau=sootExtinction(pos+.25*ray)+sootExtinction(pos+.75*ray);
            radiant=moment.w*exp(-.50*sqrt(d2)*tau)/(.17+d2);
        }
    } else {
        for(int k=0;k<4;k++){
            vec4 light=texelFetch(uLights,ivec2(k,0),0);
            if(light.w<.0001)continue;
            vec3 ray=light.xyz-pos;
            float d2=dot(ray,ray);
            float tau=0.;
            for(int j=0;j<4;j++)tau+=sootExtinction(pos+ray*((float(j)+.5)*.25));
            radiant+=light.w*exp(-.25*sqrt(d2)*tau)/(.17+d2);
        }
    }
    frag=vec4(radiant,sky,occupied,1.);
}
`;

const woodUniforms = `
uniform highp sampler2D uWood;
uniform ivec3 uWoodDims;
uniform bool uHasWood;
vec4 sampleWood(vec3 world) {
    vec3 p=(world+gridOffset)/gridScale*vec3(uDims)-.5;
    return sampleAtlas(uWood,p,uWoodDims,uTiles);
}
`;

const camera = `
uniform vec3 uOrigin, uAhead, uRight, uUp;
uniform float uAspect;
const float lens = .32;
`;

const volumeMaterial = `
uniform highp sampler3D uNoise;
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
vec3 temperaturePalette(float t) {
    vec3 c=mix(vec3(.021,.014,.15),vec3(.30,.038,.60),smoothstep(.03,.27,t));
    c=mix(c,vec3(1.,.065,.038),smoothstep(.25,.55,t));
    c=mix(c,vec3(1.,.7,.015),smoothstep(.52,.83,t));
    return mix(c,vec3(1.,1.,.89),smoothstep(.78,1.12,t));
}
// Planck ratios at approximate linear RGB sample wavelengths.
// Absolute intensity is normalized separately to display-referred exposure.
vec3 incandescent(float kelvin) {
    vec3 wavelengths=vec3(610.,550.,455.);
    vec3 exponent=(1.4387769e7/kelvin)/wavelengths;
    vec3 ratios=exp(exponent.x-exponent)*vec3(1.,1.679,4.329);
    return ratios;
}

`;
const surfaceFragment = optics => `#version 300 es
precision highp float;
precision highp int;
in vec2 uv;
out vec4 frag;
uniform int uPreset,uMode;
uniform float uVolumeDepth,uSceneHeight;
` + camera + common + woodUniforms + volumeMaterial +
    (optics ? '#define ENHANCED_FIRE\nuniform bool uStableSampling;\n'+optics.surfaceLighting : lightHelper) + surfaceShader() + `
void main(){
    vec2 ndc=2.*uv-1.;
    vec3 rd=normalize(uAhead+lens*(ndc.x*uAspect*uRight+ndc.y*uUp));
    float farT;
    vec3 bg=surface(uOrigin,rd,farT);
    // Eye-space depth is much smoother on the floor than ray distance.
    frag=vec4(max(bg,vec3(0.)),min(uVolumeDepth,farT*dot(rd,uAhead)));
}
`;

const detailFragment = `#version 300 es
precision highp float;
precision highp int;
out vec4 frag;
uniform highp sampler2D uRestA,uRestB;
uniform int uHasRest;
uniform float uRestBlend,uTime;
` + common + volumeMaterial + `
// Material coordinates travel with the solved velocity. Evaluate each
// noise field before blending across rest resets so details do not jump.
vec4 flameDetailAt(vec3 q) {
    vec3 warp=smoothNoise(q*vec3(5.2,3.5,5.2)+vec3(2.8,8.9,1.9)).rgb-.5;
    vec3 fineCoord=q*vec3(12.5,6.6,12.5)+warp*2.1;
    float fine=noise3(fineCoord)-.5;
    return vec4(warp,fine);
}
vec4 flameDetail(vec3 p) {
    if(uHasRest==0)return flameDetailAt(p-vec3(0.,uTime*.42,0.));
    vec4 a=vec4(0.), b=vec4(0.);
    if(uRestBlend<.999){
        vec3 q=sampleGrid(uRestA,p).rgb*gridScale-gridOffset;
        a=flameDetailAt(q);
    }
    if(uRestBlend>.001){
        vec3 q=sampleGrid(uRestB,p).rgb*gridScale-gridOffset;
        b=flameDetailAt(q);
    }
    return mix(a,b,uRestBlend);
}
void main(){
    ivec2 px=ivec2(gl_FragCoord.xy);
    ivec3 cell=ivec3(px%uDims.xy,px.x/uDims.x+(px.y/uDims.y)*uTiles);
    if(cell.z>=uDims.z){frag=vec4(0.);return;}
    vec4 state=texelFetch(uField,px,0);
    if(state.r<.044){frag=vec4(0.);return;}
    vec3 pos=(vec3(cell)+.5)/vec3(uDims)*gridScale-gridOffset;
    frag=flameDetail(pos)*smoothstep(.044,.17,state.r);
}`;

// Carry material samples between neighboring ray segments. The integral of
// the narrow Gaussian transfer across a segment catches bright sheets even
// when both endpoints lie on opposite sides of the luminous surface.
const enhancedSampling = `
uniform highp sampler2D uRestA,uRestB;
uniform bool uHasRest,uStableSampling;
uniform float uRestBlend;
vec4 materialNoise(vec3 q,float footprint) {
    vec3 warp=smoothNoise(q*vec3(4.5,2.5,4.5)+vec3(3.7,8.9,1.9)).rgb-.5;
    vec3 pos=q*vec3(10.7,4.8,10.7)+3.3*warp;
    vec4 eddy=smoothNoise(pos);
    float fineWeight=1.-smoothstep(.4,1.,footprint*22.9);
    float fine=(noise3(pos*2.14+1.2*eddy.gbr)-.5)*fineWeight;
    return vec4(warp,(eddy.r-.5)*1.45+fine*.48);
}
struct FlamePoint {
    vec4 state;
    float coordinate;
    float variation;
};
FlamePoint flamePoint(vec3 p,float footprint) {
    vec4 material;
    if(uHasRest){
        vec3 a=sampleGrid(uRestA,p).rgb*gridScale-gridOffset;
        vec3 b=sampleGrid(uRestB,p).rgb*gridScale-gridOffset;
        material=mix(materialNoise(a,footprint),materialNoise(b,footprint),uRestBlend);
    }else{
        vec3 velocity=sampleGrid(uVelocity,p).xyz;
        material=materialNoise(p-clamp(velocity,vec3(-2.),vec3(2.))*.11,footprint);
    }
    vec3 displaced=p+material.rgb*(.073+.10*smoothstep(.1,1.2,p.y));
    vec4 state=sampleGrid(uField,displaced);
    float mixture=max(0.,state.b)/(.018+.24*max(0.,state.a)+max(0.,state.b));
    return FlamePoint(state,.66*state.r+.72*mixture-.70+.52*material.w,material.w);
}
float erfApprox(float x) {
    // Abramowitz and Stegun, 7.1.26.
    float t=1./(1.+.3275911*abs(x));
    float poly=(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-.284496736)*t+.254829592)*t;
    return sign(x)*(1.-poly*exp(-x*x));
}
float sheetAverage(float a,float b) {
    float delta=b-a;
    if(abs(delta)<.005){float center=.5*(a+b);return exp2(-91.*center*center);}
    const float scale=7.942039; // sqrt(91 ln(2))
    return clamp(.886226925*(erfApprox(scale*b)-erfApprox(scale*a))/(scale*delta),0.,1.);
}
`;

// Resolve the expensive material coordinates, spectrum and local optical
// coefficients in world space once per solver update. Store the signed flame
// coordinate before applying its narrow transfer function so the camera can
// integrate crossings that fall between voxels.
const opticalCacheFragment=optics=>lightingHeader+volumeMaterial+optics.gasOptics+enhancedSampling+`
layout(location=0) out vec4 coefficients;
layout(location=1) out vec4 emission;
uniform ivec3 uOpticalDims,uLightDims;
uniform highp sampler2D uSmokeLighting;
uniform int uPreset,uLightTiles;
vec3 radiance(FlamePoint point, float heat, float shell, float sheet, float rate) {
    float temperature=clamp(.91*heat+.26*point.variation+.085*sheet,0.,1.07);
    if(uSharedLighting)return gasEmission(point.state,temperature,shell,sheet,uPreset,rate);
    vec3 color=vec3(1.,.040+.295*pow(temperature,1.6),.0009+.029*pow(temperature,4.));
    return color*exp2(8.65*(temperature-.69))*shell*(2.05+3.4*point.state.g);
}
void main(){
    ivec2 px=ivec2(gl_FragCoord.xy);
    ivec3 c=ivec3(px%uOpticalDims.xy,px.x/uOpticalDims.x+(px.y/uOpticalDims.y)*uTiles);
    coefficients=vec4(-.7,0.,0.,0.);
    emission=vec4(0.);
    if(c.z>=uOpticalDims.z)return;
    // Reuse the halo-expanded occupancy used by the marcher. One light cell
    // covers exactly six cached voxels in each dimension.
    ivec3 block=c/6;
    ivec2 tile=ivec2(block.z%uLightTiles,block.z/uLightTiles)*uLightDims.xy;
    if(texelFetch(uSmokeLighting,tile+block.xy,0).a<.5)return;
    vec3 p=(vec3(c)+.5)*gridScale/vec3(uOpticalDims)-gridOffset;
    vec4 raw=sampleGrid(uField,p);
    if(raw.r+raw.g<.0045)return;
    float fade=(1.-smoothstep(2.04,2.37,p.y))*(1.-smoothstep(.88,1.,max(abs(p.x),abs(p.z))));
    float smoke=max(0.,raw.g)*fade;
    coefficients.g=uSharedLighting?sootSigma(smoke):4.644*smoke;
    if(raw.r<.045)return;
    FlamePoint point=flamePoint(p,0.);
    vec4 state=point.state;
    float mixture=max(0.,state.b)/(.018+.24*max(0.,state.a)+max(0.,state.b));
    float hot=smoothstep(.17,.50,state.r+.13*point.variation);
    float shell=hot*(1.-.73*smoothstep(.36,.68,mixture));
    float rate=resolvedBurn(state,p,uPreset);
    vec3 peak=radiance(point,state.r,shell*2.168,1.,rate);
    vec3 shoulder=radiance(point,state.r,shell*.068,0.,rate);
    if(!uSharedLighting && (uPreset==2 || uPreset==7)){
        vec3 blue=vec3(.035,.30,1.5)*1.5*exp(-14.*max(0.,p.y-.075))*hot;
        shoulder+=blue;peak+=blue;
    }
    const vec3 luma=vec3(.2126,.7152,.0722);
    float peakLuma=dot(peak,luma);
    coefficients.r=point.coordinate;
    // The resolved soot carries most broadband opacity. Add only the
    // smaller absorption of hot material in the thin reaction zone.
    float frontExtinction=uSharedLighting?.36+.81*max(0.,state.g):1.1+2.*max(0.,state.g);
    coefficients.b=shell*frontExtinction*fade;
    // A clean gas reaction can emit blue even when the soot continuum is weak.
    coefficients.a=peakLuma*fade;
    emission=vec4(peak*fade,clamp(dot(shoulder,luma)/max(peakLuma,.000001),0.,1.));
}
`;

const volumeFragment = (optics,cacheTiles=0) => `#version 300 es
precision highp float;
precision highp int;
precision highp sampler3D;
in vec2 uv;
out vec4 frag;
uniform highp sampler2D uSurface;
uniform highp sampler2D uDetail,uSmokeLighting;
uniform ivec3 uLightDims;
uniform int uLightTiles;
uniform int uPreset, uMode, uSteps;
` + camera + common + volumeMaterial +
    (optics ? '#define ENHANCED_FIRE\n'+(cacheTiles?'#define CACHED_OPTICS\n':'')+optics.gasOptics+enhancedSampling : '') + `
uniform highp sampler2D uLights;
#ifdef CACHED_OPTICS
uniform highp sampler2D uCoefficients,uEmission;
uniform ivec3 uOpticalDims;
vec4 opticalSample(highp sampler2D tex,vec3 p){
    return sampleAtlas(tex,(p+gridOffset)*vec3(uOpticalDims)/gridScale-.5,uOpticalDims,${cacheTiles||1});
}
#endif
vec3 boxMin=vec3(-1.,0.,-1.), boxMax=vec3(1.,2.4,1.);
vec2 interval(vec3 ro, vec3 rd) {
    bvec3 parallel=lessThan(abs(rd),vec3(.000001));
    if(any(bvec3(parallel.x && (ro.x<boxMin.x || ro.x>boxMax.x),
                 parallel.y && (ro.y<boxMin.y || ro.y>boxMax.y),
                 parallel.z && (ro.z<boxMin.z || ro.z>boxMax.z))))return vec2(0.);
    vec3 inv=sign(rd)/max(abs(rd),vec3(.000001));
    vec3 t0=(boxMin-ro)*inv, t1=(boxMax-ro)*inv;
    vec3 l=mix(min(t0,t1),vec3(-1e6),parallel);
    vec3 h=mix(max(t0,t1),vec3(1e6),parallel);
    return vec2(max(0.,max(l.x,max(l.y,l.z))),min(h.x,min(h.y,h.z)));
}

float reactionCoordinate(vec4 state){
    float mixture=max(0.,state.b)/(.020+.265*max(0.,state.a)+max(0.,state.b));
    return .66*state.r+.73*mixture-.70;
}
float sootCoefficient(float carbon){
    return 4.8*carbon*(.46+.54*sqrt(carbon+.02));
}
vec4 volumeLight(vec3 ro,vec3 rd,float farT,vec3 emitter,float luminous) {
    vec2 hit=interval(ro,rd);
    hit.y=min(hit.y,farT);
    if(hit.y<=hit.x) return vec4(0.,0.,0.,1.);
    // The distance lattice is independent of heat and the skip mask. A
    // stationary, nonperiodic ray phase prevents coherent marching contours.
    // The pattern never rotates with simulation time.
    float ds=2./float(uSteps);
    uvec2 pixel=uvec2(gl_FragCoord.xy);
    uint bits=pixel.x*0x8da6b343u ^ pixel.y*0xd8163841u;
    bits^=bits>>16u;bits*=0x7feb352du;bits^=bits>>15u;bits*=0x846ca68bu;bits^=bits>>16u;
    float phase=float(bits&65535u)/65535.-.5;
    float anchor=dot(vec3(0.,1.2,0.)-ro,rd)+ds*.44*phase;
    float first=anchor+floor((hit.x-anchor)/ds)*ds;
    vec3 inv=sign(rd)/max(abs(rd),vec3(.000001));
    vec3 blockScale=vec3(uLightDims)/gridScale;
    vec3 face=step(vec3(0.),rd);
#ifdef ENHANCED_FIRE
#ifdef CACHED_OPTICS
    vec4 previous;
#else
    FlamePoint previous;
#endif
    int previousStep=-2;
    float footprint=2.*lens*max(hit.x,1.)/float(textureSize(uSurface,0).y);
#endif
    float pos=first;
    float macroEnd=first;
    vec3 sum=vec3(0.);
    float tr=1.;
    for(int i=0;i<384;i++){
        float begin=max(pos,hit.x);
        float end=min(pos+ds,hit.y);
        if(begin>=hit.y || tr<.012)break;
        if(end<=begin){pos+=ds;continue;}
        float middle=.5*(begin+end);
        vec3 p=ro+rd*middle;
        if(uMode!=3 && middle>=macroEnd) {
            ivec3 block=clamp(ivec3(floor((p+gridOffset)*blockScale)),ivec3(0),uLightDims-1);
            ivec2 tile=ivec2(block.z%uLightTiles,block.z/uLightTiles)*uLightDims.xy;
            vec3 exitPoint=(vec3(block)+face)/blockScale-gridOffset;
            vec3 toExit=mix(vec3(1e6),(exitPoint-p)*inv,greaterThan(abs(rd),vec3(.000001)));
            macroEnd=middle+max(.00002,min(toExit.x,min(toExit.y,toExit.z))+.00002);
            vec4 blockState=texelFetch(uSmokeLighting,tile+block.xy,0);
#ifdef ENHANCED_FIRE
            bool emptyBlock=blockState.a<.5;
#else
            bool emptyBlock=blockState.b<.5;
#endif
            if(emptyBlock){
                // Skip complete steps, but keep the final clipped step: its
                // midpoint may lie in another cell even when its start does not.
                float omitted=max(1.,ceil((macroEnd-middle)/ds));
                pos+=min(omitted,max(1.,floor((hit.y-pos)/ds)))*ds;
                continue;
            }
        }
        pos+=ds;
#ifdef CACHED_OPTICS
        vec4 left,right;
        if(uStableSampling){
            if(previousStep==i-1)left=previous;
            else left=opticalSample(uCoefficients,ro+rd*begin);
            right=opticalSample(uCoefficients,ro+rd*end);
            previous=right;previousStep=i;
        }else{
            left=opticalSample(uCoefficients,p);
            right=left;
        }
        vec4 optical=.5*(left+right);
        float k=optical.g;
        vec3 source=vec3(0.);
        if(optical.a>.00005 || optical.b>.0005){
            float sheet=sheetAverage(left.r,right.r);
            vec4 peak=opticalSample(uEmission,p);
            // The shoulders retain a substantial part of their emitted
            // energy. Squaring the transfer function outlined the folds.
            float radiantSheet=uSharedLighting?sheet*(.62+.38*sheet):sheet;
            source=peak.rgb*mix(peak.a,1.,radiantSheet);
            k+=optical.b*(.068+2.10*sheet);
        }
        if(optical.g>.003){
            vec4 lighting=sampleAtlas(uSmokeLighting,(p+gridOffset)*blockScale-.5,uLightDims,uLightTiles);
            vec3 lightRay=emitter-p;
            float cosine=dot(lightRay,rd)*inversesqrt(max(.000001,dot(lightRay,lightRay)));
            float phaseBase=max(.16,1.185-.82*cosine);
            float phase=.815*inversesqrt(phaseBase)/phaseBase;
            source+=optical.g*smokeInscatter(lighting,.61+.39*phase);
        }
#else
        vec4 state=sampleGrid(uField,p);
        float heat=state.r,soot=max(0.,state.g),fuel=max(0.,state.b);
        if(heat+soot<.0045 && uMode!=3)continue;
        vec3 source=vec3(0.);
        float k=0.;
        if(uMode==0) {
            float edgeFade=(1.-smoothstep(2.04,2.37,p.y))*(1.-smoothstep(.88,1.,max(abs(p.x),abs(p.z))));
            // Smoke keeps its solved density. Fine material motion is
            // applied to hot gas only and bounded to roughly one cell.
            float carbon=soot*.86*edgeFade;
#ifdef ENHANCED_FIRE
            float smokeK=uSharedLighting?sootSigma(soot*edgeFade):5.4*carbon;
            k=smokeK;
            if(heat>.075){
                FlamePoint left;
                FlamePoint right;
                if(uStableSampling){
                    if(previousStep==i-1)left=previous;
                    else left=flamePoint(ro+rd*begin,footprint);
                    right=flamePoint(ro+rd*end,footprint);
                    previous=right;
                    previousStep=i;
                }else{
                    left=flamePoint(p,0.);
                    right=left;
                }
                vec4 state=.5*(left.state+right.state);
                float variation=.5*(left.variation+right.variation);
                float sheet=sheetAverage(left.coordinate,right.coordinate);
                float mixture=max(0.,state.b)/(.018+.24*max(0.,state.a)+max(0.,state.b));
                float hot=smoothstep(.17,.50,state.r+.13*variation);
                float gas=hot*(1.-.73*smoothstep(.36,.68,mixture))*(.068+2.10*sheet);
                float flameT=clamp(.91*state.r+.26*variation+.085*sheet,0.,1.07);
                if(uSharedLighting){
                    float burnRate=resolvedBurn(state,p,uPreset);
                    source=gasEmission(state,flameT,gas,sheet,uPreset,burnRate)*edgeFade;
                }else{
                    vec3 color=vec3(1.,.040+.295*pow(flameT,1.6),.0009+.029*pow(flameT,4.));
                    source=color*exp2(8.65*(flameT-.69))*gas*(2.05+3.4*state.g)*edgeFade;
                    if(uPreset==2 || uPreset==7)source+=vec3(.035,.30,1.5)*1.5*exp(-14.*max(0.,p.y-.075))*hot;
                }
                float frontExtinction=uSharedLighting?.36+.81*max(0.,state.g):1.1+2.*max(0.,state.g);
                k+=gas*frontExtinction*edgeFade;
            }
#else
            float smokeK=sootCoefficient(carbon);
            k=smokeK;
            if(heat>.115 && fuel+soot>.00018) {
                vec4 detail=sampleGrid(uDetail,p);
                vec3 displacement=(.025+.038*smoothstep(.08,1.35,p.y))*detail.rgb;
                vec4 hotGas=sampleGrid(uField,p+displacement);
                float variation=.82*detail.x+.62*detail.w;
                float rawReaction=reactionCoordinate(hotGas);
                float center=rawReaction+.38*variation;
                // Approximate the transfer integral for a local linear
                // ramp rather than evaluating an arbitrarily thin sheet
                // at a single point. The area of the band is preserved.
                float slope=abs(rawReaction-reactionCoordinate(state))/max(.010,length(displacement));
                float spread=min(ds*slope,.25);
                float width=1.+5.5*spread*spread;
                float sheet=inversesqrt(width)*exp2(-56.*center*center/width);

                heat=hotGas.r;
                fuel=max(0.,hotGas.b);
                float oxygen=max(0.,hotGas.a);
                float mixture=fuel/(.020+.265*oxygen+fuel);
                float warm=smoothstep(.166,.49,heat+.101*variation);
                float reactant=smoothstep(.0001,.015,fuel);
                float richCore=1.-.69*smoothstep(.32,.65,mixture);
                float shell=warm*richCore*(.059+1.96*sheet)*mix(.63,1.,reactant);
                k+=shell*(.87+1.9*carbon);
                float thermal=clamp(.935*heat+.224*variation+.073*sheet,0.,1.09);
                float power=exp2(8.5*(thermal-.697));
                float strength=power*shell*(2.03+2.5*carbon)*edgeFade;
                float t2=thermal*thermal;
                // Warm continuum is intentionally stronger in the amber
                // part of the spectrum than at the saturated core.
                vec3 color=vec3(1.,.055+.264*thermal*sqrt(thermal),.0014+.033*t2*t2);
                source=color*strength;
                // Hot inner folds are on the richer side of the luminous
                // reaction zone. Keep the much larger orange shell thin.
                float hotSide=center-.275;
                float inner=exp2(-64.*hotSide*hotSide/width);
                float core=smoothstep(.72,1.095,heat)*smoothstep(.024,.29,oxygen);
                core*=1.-smoothstep(.57,1.28,p.y);
                source+=vec3(1.,.58,.16)*(4.6*edgeFade*power*inner*core*reactant);
                if(uPreset==2) {
                    float attachment=exp(-14.*max(0.,p.y-.075))*warm;
                    source+=vec3(.03,.20,1.20)*attachment*(1.2+.31*sheet);
                }
            }
#endif
            if(smokeK>.003) {
                vec3 cache=(p+gridOffset)*blockScale-.5;
                vec4 light=sampleAtlas(uSmokeLighting,cache,uLightDims,uLightTiles);
                vec3 lp=emitter-p;
                float cosine=dot(normalize(lp+vec3(.0001)),rd);
                float phaseBase=max(.16,1.185-.82*cosine);
                float phase=.815*inversesqrt(phaseBase)/phaseBase;
#ifdef ENHANCED_FIRE
                source+=smokeK*smokeInscatter(light,.61+.39*phase);
#else
                vec3 skylight=vec3(.043,.051,.063)*mix(.40,1.,light.g);
                vec3 fireLight=luminous>.0001 ? vec3(.176,.060,.024)*light.r*(.61+.39*phase) : vec3(0.);
                source+=smokeK*(skylight+fireLight);
#endif
            }
        }else if(uMode==1){
            float denom=smoothstep(.018,.88,heat);
            k=denom*8.;
            source=temperaturePalette(heat)*k*1.35;
        }else if(uMode==2){
            k=soot*14.;
            source=k*mix(vec3(.18,.21,.25),vec3(.93,.90,.86),exp(-1.85*soot));
        }else{
            vec3 v=sampleGrid(uVelocity,p).xyz;
            float speed=length(v);
            k=smoothstep(.18,1.65,speed)*(2.2+smoothstep(.04,.42,heat)*1.8);
            vec3 dir=v/max(speed,.001);
            vec3 c=mix(vec3(.012,.09,.30),vec3(.02,.55,.56),smoothstep(.23,1.04,speed));
            c=mix(c,vec3(.89,.97,.33),smoothstep(1.0,2.25,speed));
            source=k*c*(.8+.2*abs(dir.y));
        }
#endif
        float segment=end-begin;
        float tau=k*segment;
        // The bounded march samples have short optical lengths. The
        // rational approximation matches Beer–Lambert to second order
        // and shares one reciprocal with the integrated source.
        float weight=1./(1.+.5*tau);
        float opacity=tau*weight;
        sum+=tr*source*(segment*weight);
        tr*=max(0.,1.-opacity);
    }
    return vec4(sum,tr);
}
void main(){
    vec2 ndc=2.*uv-1.;
    vec3 rd=normalize(uAhead+lens*(ndc.x*uAspect*uRight+ndc.y*uUp));
    vec4 moment=vec4(0.);
    for(int i=0;i<4;i++){
        vec4 light=texelFetch(uLights,ivec2(i,0),0);
        moment+=vec4(light.xyz*light.w,light.w);
    }
    vec3 emitter=moment.xyz/max(moment.w,.00001);
    ivec2 surfacePixel=ivec2(uv*vec2(textureSize(uSurface,0)));
    float farT=texelFetch(uSurface,surfacePixel,0).a/dot(rd,uAhead);
    frag=volumeLight(uOrigin,rd,farT,emitter,min(2.,moment.w));
}`;

const resolveFragment = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 frag;
uniform highp sampler2D uSurface,uVolume;
void main(){
    vec4 surface=texelFetch(uSurface,ivec2(gl_FragCoord.xy),0);
    vec2 size=vec2(textureSize(uVolume,0));
    vec2 f=uv*size-.5;
    ivec2 lower=ivec2(floor(f));
    ivec2 maxIndex=ivec2(size)-1;
    vec2 fraction=fract(f);
    vec4 sum=vec4(0.);
    float total=0.;
    vec2 surfaceSize=vec2(textureSize(uSurface,0));
    // Reconstruct premultiplied radiance and transmittance in HDR. Depth
    // suppresses tap contributions from the gas behind the solid pixels.
    for(int i=0;i<4;i++){
        ivec2 tap=ivec2(i&1,i>>1);
        ivec2 q=clamp(lower+tap,ivec2(0),maxIndex);
        ivec2 guide=min(ivec2((vec2(q)+.5)/size*surfaceSize),ivec2(surfaceSize)-1);
        float depth=texelFetch(uSurface,guide,0).a;
        vec2 spatial=mix(1.-fraction,fraction,vec2(tap));
        float relative=abs(depth-surface.a)/(.055+.065*min(depth,surface.a));
        float weight=spatial.x*spatial.y*exp2(-4.*relative*relative);
        sum+=texelFetch(uVolume,q,0)*weight;
        total+=weight;
    }
    vec4 gas=(sum+vec4(0.,0.,0.,.0003))/(total+.0003);
    frag=vec4(max(gas.rgb+gas.a*surface.rgb,vec3(0.)),1.);
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
    woodInfo=vec3(seed,face,min(radii.x,radii.y));
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

// Integrate each char seam across one projected pixel. Distant fine cracks
// converge to their mean coverage instead of alternating black and orange.
float sfBandIntegral(float t,float duty) {
    return floor(t)*duty+min(fract(t),duty);
}
float sfBand(float phase,float duty,float span) {
    float width=max(.034,span);
    float center=phase+.5*duty;
    float cover=(sfBandIntegral(center+.5*width,duty)-sfBandIntegral(center-.5*width,duty))/width;
    return mix(clamp(cover,0.,1.),duty,smoothstep(.75,1.65,width));
}
float sfFilteredNoise(vec3 p,float footprint) {
    if(footprint>=1.20)return .5;
    return .5+(noise3(p)-.5)*(1.-smoothstep(.40,1.20,footprint));
}

#ifdef ENHANCED_FIRE
vec3 sfNightSky(vec3 rd) {
    float horizon=1.-clamp(2.8*abs(rd.y+.11),0.,1.);
    float skyNorm=horizon*horizon;
    return mix(vec3(.011,.014,.018),vec3(.024,.029,.036),skyNorm);
}

float sfLogContact(vec2 p,vec2 a,vec2 b){
    vec2 axis=b-a,path=p-a;
    vec2 nearest=path-axis*clamp(dot(path,axis)/dot(axis,axis),0.,1.);
    return smoothstep(.00052,.012,dot(nearest,nearest));
}

float sfBedContact(vec2 p){
    float cap=sfLogContact(p,vec2(-.39,-.11),vec2(.30,-.17));
    cap=min(cap,sfLogContact(p,vec2(-.32,.18),vec2(.39,.085)));
    cap=min(cap,sfLogContact(p,vec2(-.28,-.31),vec2(.17,.30)));
    return mix(.21,1.,cap);
}

// Locally rounded pieces of gravel in the dirt. The footprint fades both
// color and tangent-space relief as the stones get smaller than a pixel.
float sfGravel(vec2 p,float footprint,out vec3 normal,out float contact) {
    vec2 cell=floor(p*7.7);
    vec3 seed=fract(vec3(cell.xyx)*vec3(.1031,.0973,.1099));
    seed+=dot(seed,seed.yzx+19.19);
    seed=fract((seed.xxy+seed.yzz)*seed.zyx);
    vec2 shift=.66*(seed.xy-.5);
    vec2 radial=fract(p*7.7)-.5-shift;
    float radius=min(mix(.12,.30,seed.z),.475-max(abs(shift.x),abs(shift.y)));
    float stretch=mix(.68,1.0,fract(seed.x*37.2));
    vec2 longAxis=normalize(vec2(.12+seed.y,seed.x-.52));
    float distance=length(vec2(dot(radial,longAxis)*stretch,dot(radial,vec2(-longAxis.y,longAxis.x))))/radius;
    float scale=1.-smoothstep(.010,.032,footprint);
    // A density variation and missing cells break any underlying lattice.
    scale*=smoothstep(.15,.43,fract(seed.x*8.37+seed.y*19.42+seed.z*4.71));
    float stone=(1.-smoothstep(.83,1.02,distance))*scale;
    normal=normalize(vec3(radial.x*2.8*stone,1.,radial.y*2.8*stone));
    contact=1.-.13*scale*smoothstep(.79,.98,distance)*(1.-smoothstep(1.02,1.34,distance));
    return stone;
}
#endif

vec3 surface(vec3 ro,vec3 rd,out float hitT) {
    hitT=1000.;
    vec3 ambient=vec3(.010,.013,.016);
    bool wood=(uPreset==0 || uPreset==3);
    bool narrow=uPreset==2 || uPreset==7;
#ifdef ENHANCED_FIRE
    bool emberBed=uPreset==6;
    bool hasBurner=uPreset==1 || narrow;
    if(uMode==0)ambient=sfNightSky(rd);
#else
    bool hasBurner=!wood;
#endif
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
        } else if(hasBurner) {
            float r=narrow ? .138 : .405;
            sfCylinder(ro,rd,r*.91,.002,.052,hitT,norm,part,1.);
            sfCylinder(ro,rd,r,.050,.074,hitT,norm,part,3.);
        }
    }
    if(hitT>999.) return ambient*.60;
    vec3 p = ro+rd*hitT;
    if (length(p.xz)>12.) return ambient*.60;
    // At grazing angles one pixel covers more of the tangent plane. A
    // geometric footprint remains finite and independent of neighboring hit IDs.
    float worldPixel=2.*lens*hitT/(uSceneHeight*max(.075,abs(dot(norm,-rd))));
#ifdef ENHANCED_FIRE
    if(!uStableSampling)worldPixel=0.;
#endif
    vec3 cool=vec3(.10,.132,.16)*(.48+.52*max(0.,dot(norm,normalize(vec3(-.6,.8,-.4)))));

    if(info.x>0.) {
        float seed=info.x;
        float angle=atan(local.y,local.x)/6.2831853;
        float s=local.z;
        float ringDist=length(local.xy+vec2(.08*sin(seed),.10*cos(seed)));
        // Long fissures follow the grain. Broken crosscuts and displaced
        // phases interrupt the repeating pattern around cylindrical bark.
        float localPixel=worldPixel/info.z;
        vec3 noisePoint=vec3(local.xy*1.35,s*4.5+seed*9.2);
        float broad=sfFilteredNoise(noisePoint,1.35*localPixel);
        float mid=sfFilteredNoise(noisePoint*2.6,3.51*localPixel);
        float fine=sfFilteredNoise(noisePoint*6.5,8.78*localPixel);
        float axialPhase=angle*9.+2.35*broad+.46*mid+seed;
        float crossPhase=s*6.2+1.30*mid+.72*broad+seed;
#ifdef ENHANCED_FIRE
        // Intermittent raised grain and curled scales interrupt the regular
        // bevel normals without adding smaller, aliasing geometry.
        vec3 surfaceNormal=norm;
        if(info.y<.5 && localPixel<.45){
            vec3 tangent=abs(norm.y)<.995?normalize(cross(vec3(0.,1.,0.),norm)):vec3(1.,0.,0.);
            vec3 grain=vec3(norm.z*.44,.45,-norm.x*.44);
            float rift=sin(6.2831853*axialPhase)*(1.-smoothstep(.09,.45,localPixel));
            surfaceNormal=normalize(norm+tangent*(.19*rift+.24*(mid-.5))+grain*(.34*(fine-.5)));
        }
#else
        vec3 surfaceNormal=norm;
#endif
        SfIrradiance incident=sfIncident(p,surfaceNormal,-rd,wood,seed);
        float hotBed=smoothstep(.001,.39,incident.flux);
        float axial=sfBand(axialPhase,.115,localPixel*1.85);
        float cross=sfBand(crossPhase,.092,worldPixel*8.+localPixel*.45);
        cross*=smoothstep(.27,.67,broad+.20*(mid-.5));
        float split=axial+cross-axial*cross;
        float warmth=clamp(1.12-length(p.xz)*1.95+.52*(broad-.5),.10,1.);
        float scale=.44+.56*(1.-split);
        float ridges=smoothstep(.32,.77,broad*.55+mid*.34+fine*.11);
        vec3 albedo=mix(vec3(.058,.036,.025),vec3(.22,.158,.107),ridges)*scale;
        float hotPockets=smoothstep(.32,.72,broad*.32+mid*.55+fine*.13);
        float embers=(.015+.095*hotPockets+.88*split)*pow(warmth,1.9)*(.12+.88*hotPockets);
        float ash=smoothstep(.64,.82,mid*.48+broad*.34+fine*.18);
        albedo+=vec3(.17,.16,.15)*ash*scale;
        if(info.y>.5) {
            float rings=sfBand(ringDist*4.3+broad*.50,.18,localPixel*4.6);
            float radial=sfBand(angle*5.+broad*.55+seed,.10,localPixel*.88/max(.25,ringDist));
            albedo=mix(vec3(.078,.044,.025),vec3(.26,.135,.064),.18+.68*broad)*(1.-.32*rings);
            embers=warmth*.68*radial*smoothstep(.10,.38,ringDist);
            albedo*=1.-.45*radial;
        }
        float emberGlow=hotBed;
        float coalHeat=.60;
        if(uHasWood) {
            float gridLength=length(norm*gridScale/vec3(uDims));
            vec4 solid=sampleWood(p-norm*gridLength*.63);
            vec4 deep=sampleWood(p-norm*gridLength*1.22);
            if(deep.g+deep.b>solid.g+solid.b)solid=deep;
            coalHeat=solid.r;
            if(uMode==4) {
                float marks=1.-.36*split;
                return marks*(.043+temperaturePalette(coalHeat)*smoothstep(.018,.40,coalHeat)*.88);
            }
            float charFraction=smoothstep(.028,.115,solid.b);
            float redHeat=smoothstep(.28,.96,coalHeat);
            emberGlow=1.72*redHeat*redHeat*(.12+.88*charFraction);
            albedo*=1.-.53*charFraction;
        }
#ifdef ENHANCED_FIRE
        if(uSharedLighting && !uHasWood){
            // A gas-only solver has no stored solid thermal history. Estimate the
            // contact temperature from air just outside the rendered facet.
            float cell=length(norm*gridScale/vec3(uDims));
            float localHeat=sampleGrid(uField,p+norm*cell*.80).r;
            coalHeat=mix(.30*hotBed,localHeat,.78);
            emberGlow*=.36+.82*smoothstep(.12,.82,coalHeat);
        }
        if(uSharedLighting){
            // Char platelets absorb most incident firelight. Keep their ash and
            // sheltered, glowing fissures distinct at the same image footprint.
            float charred=smoothstep(.12,.64,coalHeat);
            float flakyAsh=ash*smoothstep(.17,.68,broad+.28*(fine-.5));
            albedo*=mix(vec3(.93,.85,.77),vec3(.29,.28,.28),charred*(1.-.82*flakyAsh));
            albedo+=vec3(.11,.105,.097)*flakyAsh*charred;
            embers*=mix(.32,1.18,smoothstep(.25,.78,.72*mid+.28*broad));
        }
        vec3 emberColor=uSharedLighting?thermalColor(900.+460.*clamp(coalHeat,0.,1.4)):incandescent(860.+430.*clamp(coalHeat,0.,1.4));
#else
        vec3 emberColor=incandescent(860.+430.*clamp(coalHeat,0.,1.4));
#endif
        vec3 coal=emberColor*2.5*emberGlow*embers*(uMode==0 ? 1. : .11);
        // Keep the cold log silhouettes visible enough to choose an ignition point.
        vec3 woodFill=vec3(.043,.057,.075)*(.37+.63*max(0.,norm.y))*(1.-.58*hotBed);
        vec3 shade=albedo*(cool+woodFill+.08*incident.fill+.93*incident.direct);
#ifdef ENHANCED_FIRE
        if(uSharedLighting) {
            vec3 bounce=vec3(.11,.048,.022)*hotBed*(.58+.42*(1.-surfaceNormal.y));
            shade=albedo*(cool+woodFill+bounce+.045*incident.fill+.81*incident.direct);
        }
#endif
        return shade+coal;
    }
    if (part>0.) {
        SfIrradiance incident=sfIncident(p,norm,-rd,false,0.);
        float r=narrow ? .138 : .405;
        float rr=length(p.xz)/r;
        float grain=sfFilteredNoise(p*193.,worldPixel*193.);
        bool rim=rr>.927 || part==3.;
        vec3 metal=mix(vec3(.045,.042,.039),vec3(.115,.100,.083),grain);
        vec3 steel=metal*(cool+.047*incident.fill+.51*incident.direct)+.31*incident.specular;
        if(rim) return steel+vec3(.036,.023,.015)*(1.-smoothstep(.010,.045,abs(rr-.966)));
        if(narrow) {
            vec2 cell=fract(p.xz*59.+.5)-.5;
            float span=worldPixel*59.;
            float edge=max(.045,.48*span);
            float open=1.-smoothstep(.26-edge,.26+edge,length(cell));
            open=mix(open,.212,smoothstep(.40,.93,span));
            return mix(steel*.70,vec3(.018,.006,.003),open);
        }
        return vec3(.011,.009,.008)+.096*incident.gloss*(.56+.44*grain);
    }
    float r=length(p.xz);
    if(r>12.) return ambient*.60;
    float broad=sfFilteredNoise(vec3(p.xz*5.1,2.),5.1*worldPixel);
    float mid=sfFilteredNoise(vec3(p.xz*29.,8.),29.*worldPixel);
    float grains=sfFilteredNoise(vec3(p.xz*145.,18.),145.*worldPixel);
    float detail=.34*broad+.31*mid+.35*grains;
    vec3 earth=mix(vec3(.043,.043,.045),vec3(.122,.115,.105),detail);
    float soot=wood ? 1.-smoothstep(.27,.77,r+.12*(broad-.5)) : 0.;
    earth*=1.-.48*soot;
    float ash=wood ? (1.-smoothstep(.30,.64,r+.12*broad))*smoothstep(.43,.83,mid*.57+grains*.43) : 0.;
    earth+=vec3(.095,.088,.075)*ash;
    float contact=wood ? mix(.33,1.,smoothstep(.12,.46,r)) : mix(.26,1.,smoothstep(.28,.70,r));
#ifdef ENHANCED_FIRE
    vec3 groundNormal=norm;
    if(r<2.2 && uMode==0 && worldPixel<.032){
        float stoneContact;
        vec2 gravelPos=p.xz+.11*(vec2(broad,.44*mid+.56*broad)-.5);
        float stone=sfGravel(gravelPos,worldPixel,groundNormal,stoneContact)*(1.-smoothstep(1.6,2.2,r));
        groundNormal=mix(norm,groundNormal,1.-smoothstep(1.6,2.2,r));
        earth=mix(earth*stoneContact,vec3(.126,.108,.086)*mix(.44,1.32,grains),stone*.69);
    }
    SfIrradiance incident=sfIncident(p,groundNormal,-rd,wood,-1.);
    if(emberBed && uMode==0){
        float radius=length(p.xz/vec2(.49,.32));
        float bed=1.-smoothstep(.69,1.04,radius+.23*(broad-.5));
        if(bed>.001){
            vec4 nearCoal=sampleGrid(uField,vec3(p.x,.076,p.z));
            float heat=max(nearCoal.r,.75*sampleGrid(uField,vec3(p.x,.126,p.z)).r);
            float bright=smoothstep(.19,.68,.56*grains+.44*mid);
            float crevice=1.-smoothstep(.035,.20,abs(mid+.38*(grains-.5)-.53));
            float heatResponse=smoothstep(.035,.46,heat);
            float glow=bed*heatResponse*(.24+.76*crevice)*bright;
            float coolAsh=smoothstep(.46,.72,.43*broad+.36*mid+.21*grains);
            vec3 char=vec3(.047,.039,.035)+coolAsh*vec3(.15,.151,.152);
            earth=mix(earth,char*(.73+.56*grains),bed);
            // The charcoal radiates from below cooler ash platelets. A
            // coarser red component gives exposed pockets some depth.
            vec3 coalLight=thermalColor(1040.+540.*clamp(heat,0.,1.));
            return earth*(cool*.95+.74*incident.direct)+.84*glow*coalLight;
        }
    }
#else
    SfIrradiance incident=sfIncident(p,norm,-rd,wood,-1.);
#endif
    float hotBed=smoothstep(.001,.39,incident.flux);
    vec3 ground=earth*(cool*.74+incident.direct*mix(contact,1.,wood ? .68 : 0.)*(wood ? .83 : .62));
#ifdef ENHANCED_FIRE
    if(wood && r<.91 && uMode==0) {
        float contactShadow=sfBedContact(p.xz);
        float embankment=1.-smoothstep(.35,.82,r+.17*(broad-.5));
        float charFlakes=smoothstep(.40,.73,.45*broad+.32*mid+.23*grains);
        float ashPatches=embankment*smoothstep(.41,.75,.63*mid+.37*grains);
        vec3 charDust=mix(vec3(.028,.025,.024),vec3(.125,.112,.100),ashPatches);
        earth=mix(earth,charDust,embankment*(.59+.36*charFlakes));
        vec3 skyReflection=cool*.90*contactShadow;
        vec3 floorBounce=vec3(.065,.023,.011)*hotBed*contactShadow;
        ground=earth*(skyReflection+floorBounce+.77*incident.direct);
    }
#endif
    if(wood && r<.45 && uMode==0) {
        float bed=(1.-smoothstep(.19,.45,r))*smoothstep(.54,.77,broad*.32+mid*.45+grains*.23);
        float radiance=hotBed;
        if(uHasWood) {
            float bedHeat=max(sampleWood(vec3(p.x,.055,p.z)).r,sampleWood(vec3(p.x,.095,p.z)).r);
            radiance=smoothstep(.28,.95,bedHeat);
        }
        ground+=vec3(.88,.083,.004)*bed*radiance;
#ifdef ENHANCED_FIRE
        // Break the bright underbed into coves separated by cold chips and
        // use local simulated heat to confine glow to the reacting pile.
        if(uSharedLighting){
            float localHeat=sampleGrid(uField,vec3(p.x,.094,p.z)).r;
            float warm=smoothstep(.08,.57,localHeat);
            ground+=vec3(.58,.052,.004)*bed*warm*(.32+.68*grains);
        }
#endif
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
        vec2 d=.84/vec2(textureSize(uWide,0));
        vec3 broad=texture(uWide,uv).rgb*4.;
        broad+=texture(uWide,uv+d).rgb+texture(uWide,uv-d).rgb;
        broad+=texture(uWide,uv+d*vec2(-1,1)).rgb+texture(uWide,uv+d*vec2(1,-1)).rgb;
        c+=.052*texture(uNear,uv).rgb+.027*texture(uMid,uv).rgb+.016*broad/8.;
    }
    c*=exp2(uExposure);
    c=clamp((c*(2.51*c+.03))/(c*(2.43*c+.59)+.14),0.,1.);
    c=mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(vec3(.0031308),c));
    frag=vec4(c,1.);
}`;

const sparksUpdate = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec4 aState;
out vec4 nextState;
uniform float uDt,uTime;
` + common + woodUniforms + `
float random1(float p){return fract(sin(p*41.337+11.23)*43758.545);}
void main(){
    float id=float(gl_VertexID);
    vec4 s=aState;
    float life=.68+1.60*random1(id+4.);
    float age=s.w+uDt;
    if(age>=life || (s.w<0. && age>=0.)){
        float cycle=floor(uTime*3.);
        float a=random1(id+cycle)*6.283185;
        float r=.32*sqrt(random1(id+9.13+cycle));
        vec3 bed=vec3(cos(a)*r,.095+.052*random1(id+21.2+cycle),sin(a)*r);
        s.xyz=bed+vec3(0.,.10,0.);
        float temp=sampleGrid(uField,s.xyz).r;
        bool glowing=temp>.17;
        if(uHasWood){
            vec4 solid=sampleWood(bed);
            vec4 lower=sampleWood(bed-vec3(0.,.043,0.));
            glowing=(solid.r>.24 && solid.b>.019) || (lower.r>.24 && lower.b>.019);
        }
        if(glowing && temp>.055 && s.w<0.){
            age=0.;
        }else{
            age=-.23-random1(id+3.21+cycle)*(glowing?1.8:3.2);
            s.y=-1.;
        }
    }
    if(age>=0. && s.y>=0.){
        vec3 flow=sampleGrid(uVelocity,s.xyz).xyz;
        vec2 drift=vec2(cos(id*7.81),sin(id*7.81))*.054;
        s.xyz+=uDt*(flow*(.60+.32*random1(id+91.))+vec3(drift.x,.24-.21*age,drift.y));
    }
    nextState=vec4(s.xyz,age);
    gl_Position=vec4(0.);
}`;

const sparksVertex = `#version 300 es
precision highp float;
layout(location=0) in vec4 aState;
uniform float uHeight,uTime;
out float brightness;
out vec2 trail;
out float emberAge;
` + camera + common + `
void main(){
    float id=float(gl_VertexID);
    float seed=fract(sin(id*41.337+11.23)*43758.545);
    vec3 d=aState.xyz-uOrigin;
    float z=dot(d,uAhead);
    gl_Position=vec4(dot(d,uRight)/(lens*uAspect),dot(d,uUp)/lens,0.,z);
    float life=.68+1.60*fract(sin((id+4.)*41.337+11.23)*43758.545);
    brightness=smoothstep(0.,.11,aState.w)*(1.-smoothstep(.40*life,life,aState.w));
    if(aState.y<.13 || aState.y>2.36) brightness=0.;
    brightness*=.85+.15*sin(uTime*5.1+id);
    vec3 flow=sampleGrid(uVelocity,aState.xyz).xyz;
    vec2 travel=vec2(dot(uRight,flow),-dot(uUp,flow));
    trail=travel/max(.032,length(travel));
    emberAge=clamp(aState.w/life,0.,1.);
    gl_PointSize=clamp(uHeight*(.008+.009*seed)/max(.13,z),2.,11.);
}`;

const sparksFragment = `#version 300 es
precision highp float;
in float brightness;
in vec2 trail;
in float emberAge;
out vec4 frag;
void main() {
    vec2 p=gl_PointCoord*2.-1.;
    float along=dot(p,trail);
    float crosswise=dot(p,vec2(-trail.y,trail.x));
    float head=exp(-27.*dot(p,p));
    float wake=exp(-42.*crosswise*crosswise-5.5*along*along)*(1.-smoothstep(-.20,.14,along));
    float track=brightness*(head+.10*wake);
    vec3 radiation=mix(vec3(7.2,2.15,.43),vec3(3.4,.33,.026),emberAge);
    frag=vec4(radiation*track,0.);
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

function basis({yaw=.38,pitch=.14,distance=3.9,targetHeight=.98}={}){
    const sy=Math.sin(yaw),cy=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch);
    const ahead=[-sy*cp,-sp,-cy*cp], right=[cy,0,-sy],up=[-sy*sp,cp,-cy*sp];
    const origin=[-ahead[0]*distance,targetHeight-ahead[1]*distance,-ahead[2]*distance];
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
    constructor(gl,optics=null){
        this.gl=gl;
        this.optics=optics;
        this.programs={};
        this.targets=[];
        this.smokeLights=null;
        this.lightingState=null;
        this.detailTarget=null;
        this.opticalCache=null;
        this.detailState=null;
        this.volumeTarget=null;
        this.renderWidth=0;
        this.renderHeight=0;
        this.sparkBuffers=[];
        this.sparkVaos=[];
        this.numSparks=100;
        this.sparkIndex=0;
        this.lastTime=null;
        try{
            gl.getExtension('EXT_color_buffer_float');
            const p=this.programs;
            p.irradiance=buildProgram(gl,vertex,optics?lightingHeader+optics.irradianceBody:irradianceFragment);
            p.smokeLighting=buildProgram(gl,vertex,optics?lightingHeader+optics.smokeLightingBody:smokeLightingFragment);
            p.surface=buildProgram(gl,vertex,surfaceFragment(optics));
            if(!optics)p.volume=buildProgram(gl,vertex,volumeFragment(null));
            if(!optics)p.detail=buildProgram(gl,vertex,detailFragment);
            p.resolve=buildProgram(gl,vertex,resolveFragment);
            p.bloom=buildProgram(gl,vertex,bloomFragment);
            p.composite=buildProgram(gl,vertex,outputFragment);
            p.update=buildProgram(gl,sparksUpdate,skipColor,true);
            p.sparks=buildProgram(gl,sparksVertex,sparksFragment);
            this.noise=noiseVolume(gl);
            this.lights=this.target(4,optics?2:1);
            if(optics)this.opticsLut=optics.createOpticsLut(gl);
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
    detailBuffer(fluid){
        const [nx,ny,nz]=fluid.dims;
        const width=nx*fluid.tiles,height=ny*Math.ceil(nz/fluid.tiles);
        const target=this.detailTarget;
        if(target?.w===width && target?.h===height)return target;
        if(target){
            this.gl.deleteTexture(target.tex);
            this.gl.deleteFramebuffer(target.fb);
        }
        this.detailState=null;
        this.detailTarget=this.target(width,height);
        return this.detailTarget;
    }
    clearOpticalCache(){
        if(!this.opticalCache)return;
        const {coefficients,emission}=this.opticalCache;
        for(const buffer of [coefficients,emission]){
            this.gl.deleteFramebuffer(buffer.fb);
            this.gl.deleteTexture(buffer.tex);
        }
        this.opticalCache=null;
    }
    updateOpticalCache(fluid,options,preset,smoke){
        const gl=this.gl, programs=this.programs;
        if(!programs.bakeOptics)programs.bakeOptics=buildProgram(gl,vertex,opticalCacheFragment(this.optics));
        if(!this.opticalCache || this.opticalCache.fluid!==fluid){
            this.clearOpticalCache();
            const dims=fluid.dims.map(x=>x*1.5);
            const w=dims[0]*fluid.tiles,h=dims[1]*Math.ceil(dims[2]/fluid.tiles);
            const coefficients=this.target(w,h);
            let emission;
            try{
                emission=this.target(w,h);
                this.bindTarget(coefficients);
                gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,gl.TEXTURE_2D,emission.tex,0);
                gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
                if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)
                    throw Error('Unable to allocate fire optics targets.');
            }catch(error){
                for(const target of [coefficients,emission])if(target){
                    gl.deleteFramebuffer(target.fb);gl.deleteTexture(target.tex);
                }
                throw error;
            }
            this.opticalCache={fluid,dims,coefficients,emission,state:null};
        }
        const cache=this.opticalCache;
        if(cache.state===this.lightingState)return cache;
        const u=programs.bakeOptics;
        this.bindTarget(cache.coefficients);
        gl.useProgram(u.program);
        this.grid(u,fluid);
        this.bindOptics(u,fluid,options);
        this.texture(2,gl.TEXTURE_3D,this.noise,u.uNoise);
        this.texture(4,gl.TEXTURE_2D,fluid.restA??fluid.field,u.uRestA,this.linear);
        this.texture(5,gl.TEXTURE_2D,fluid.restB??fluid.field,u.uRestB,this.linear);
        gl.uniform1i(u.uHasRest??null,fluid.restA && fluid.restB?1:0);
        gl.uniform1f(u.uRestBlend??null,fluid.restBlend??0);
        this.texture(7,gl.TEXTURE_2D,smoke.tex,u.uSmokeLighting);
        gl.uniform3iv(u.uLightDims,smoke.dims);
        gl.uniform1i(u.uLightTiles,smoke.tiles);
        gl.uniform3iv(u.uOpticalDims,cache.dims);
        gl.uniform1i(u.uPreset??null,preset);
        gl.drawArrays(gl.TRIANGLES,0,3);
        cache.state=this.lightingState;
        return cache;
    }
    volumeBuffer(width,height){
        const gl=this.gl;
        if(this.volumeTarget?.w===width && this.volumeTarget?.h===height)return this.volumeTarget;
        if(this.volumeTarget){
            gl.deleteTexture(this.volumeTarget.tex);
            gl.deleteFramebuffer(this.volumeTarget.fb);
        }
        this.volumeTarget=this.target(width,height);
        return this.volumeTarget;
    }
    smokeCache(quality,gridDims){
        if(this.smokeLights?.quality===quality && this.smokeLights.dims[0]*4===gridDims[0])return this.smokeLights;
        const gl=this.gl;
        if(this.smokeLights){
            gl.deleteTexture(this.smokeLights.tex);
            gl.deleteFramebuffer(this.smokeLights.fb);
        }
        const dims=gridDims.map(value=>value/4);
        const tiles=Math.ceil(Math.sqrt(dims[2]));
        this.smokeLights={...this.target(dims[0]*tiles,dims[1]*Math.ceil(dims[2]/tiles)),dims,tiles,quality};
        return this.smokeLights;
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
        gl.uniform1i(location??null,unit);
    }
    grid(program,fluid,velocity=true){
        const gl=this.gl;
        this.texture(0,gl.TEXTURE_2D,fluid.field,program.uField,this.linear);
        if(velocity)this.texture(1,gl.TEXTURE_2D,fluid.velocity,program.uVelocity,this.linear);
        gl.uniform3iv(program.uDims,fluid.dims);
        gl.uniform1i(program.uTiles,fluid.tiles);
    }
    bindOptics(program,fluid,options){
        const gl=this.gl;
        this.texture(9,gl.TEXTURE_2D,this.opticsLut,program.uOptics,this.linear);
        this.texture(10,gl.TEXTURE_2D,fluid.reaction??fluid.field,program.uReaction,this.linear);
        gl.uniform1i(program.uHasReaction??null,fluid.reaction?1:0);
        gl.uniform1i(program.uSharedLighting??null,options.sharedLighting!==false?1:0);
        gl.uniform1i(program.uStableSampling??null,options.stableSampling!==false?1:0);
    }
    bindWood(program,fluid){
        const gl=this.gl, wood=fluid.wood;
        this.texture(8,gl.TEXTURE_2D,wood??fluid.field,program.uWood,this.linear);
        gl.uniform3iv(program.uWoodDims,wood?fluid.woodDims:fluid.dims);
        gl.uniform1i(program.uHasWood,wood?1:0);
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
        if(preset!==0 && preset!==3 && !(this.optics && preset===6))return;
        if(delta>0){
            const u=p.update,dest=1-this.sparkIndex;
            gl.useProgram(u.program);
            this.grid(u,fluid);
            this.bindWood(u,fluid);
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
        this.grid(u,fluid);
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
    drawSurface(target,fluid,b,aspect,options,preset,mode){
        const gl=this.gl, p=this.programs;
        this.bindTarget(target);
        gl.useProgram(p.surface.program);
        this.camera(p.surface,b,aspect);
        this.texture(2,gl.TEXTURE_3D,this.noise,p.surface.uNoise);
        this.texture(6,gl.TEXTURE_2D,this.lights.tex,p.surface.uLights,this.linear);
        this.grid(p.surface,fluid,false);
        this.bindWood(p.surface,fluid);
        if(this.optics)this.bindOptics(p.surface,fluid,options);
        gl.uniform1i(p.surface.uPreset,preset);
        gl.uniform1i(p.surface.uMode,mode);
        // All geometry behind the smoke domain is equivalent for upsampling.
        // Clamp the guide there so floor depth at the horizon cannot band.
        const [ax,ay,az]=b.ahead;
        const volumeDepth=(options.distance??3.9)+Math.abs(ax)+Math.abs(az)+(1.2-(options.targetHeight??.98))*ay+1.2*Math.abs(ay)+.05;
        gl.uniform1f(p.surface.uVolumeDepth,volumeDepth);
        gl.uniform1f(p.surface.uSceneHeight,gl.drawingBufferHeight);
        gl.drawArrays(gl.TRIANGLES,0,3);
    }
    draw(fluid,options={}){
        const gl=this.gl, w=gl.drawingBufferWidth,h=gl.drawingBufferHeight;
        if(!w || !h || gl.isContextLost())return;
        const aspect=w/h, b=basis(options);
        const mode=({beauty:0,temperature:1,smoke:2,velocity:3,wood:4})[options.mode]??0;
        const preset=presetCodes[options.preset]??0;
        const cachedQuality=options.quality==='low' || options.quality==='medium';
        const useCache=this.optics && cachedQuality && mode===0;
        if(!cachedQuality && this.opticalCache)this.clearOpticalCache();
        const time=options.time??0;
        gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);
        gl.disable(gl.SCISSOR_TEST);gl.disable(gl.RASTERIZER_DISCARD);
        gl.colorMask(true,true,true,true);
        if(!this.targets.length || this.targets[0].w!==w || this.targets[0].h!==h){
            this.clearTargets();
            for(const div of [1,3,8,18,1])this.targets.push(this.target(Math.max(1,Math.floor(w/div)),Math.max(1,Math.floor(h/div))));
        }
        const [scene,near,mid,wide,surface]=this.targets, p=this.programs;
        gl.bindVertexArray(this.vao);
        // Lighting is a function of the simulation state, never the camera.
        const pass=this.optics?options.profilePass:null;
        pass?.('lighting');
        const revision=options.stateRevision;
        const shared=Boolean(this.optics && options.sharedLighting!==false);
        if(!Number.isSafeInteger(revision) || this.lightingState?.fluid!==fluid || this.lightingState.revision!==revision || this.lightingState.shared!==shared || this.lightingState.preset!==preset){
            this.bindTarget(this.lights);
            gl.useProgram(p.irradiance.program);
            this.grid(p.irradiance,fluid,false);
            if(this.optics){
                this.bindOptics(p.irradiance,fluid,options);
                gl.uniform1i(p.irradiance.uPreset??null,preset);
            }
            gl.drawArrays(gl.TRIANGLES,0,3);
            this.lightingState={fluid,revision,shared,preset};
        }

        if(mode!==4){
            const smoke=this.smokeCache(options.quality,fluid.dims);
            const needsShading=mode===0;
            const needsSmokeUpdate=smoke.state!==this.lightingState || (needsShading && !smoke.shaded);
            if(mode<3 && needsSmokeUpdate){
                const light=p.smokeLighting;
                this.bindTarget(smoke);
                gl.useProgram(light.program);
                this.grid(light,fluid,false);
                if(this.optics)this.bindOptics(light,fluid,options);
                this.texture(6,gl.TEXTURE_2D,this.lights.tex,light.uLights);
                gl.uniform3iv(light.uLightDims,smoke.dims);
                gl.uniform1i(light.uLightTiles,smoke.tiles);
                gl.uniform1i(light.uLightProbes,options.quality==='low'?1:4);
                gl.uniform1i(light.uLightShading,needsShading?1:0);
                gl.drawArrays(gl.TRIANGLES,0,3);
                smoke.state=this.lightingState;
                smoke.shaded=needsShading;
            }
            let detailTexture=fluid.field;
            if(mode===0 && !this.optics){
                const target=this.detailBuffer(fluid);
                detailTexture=target.tex;
                if(this.detailState!==this.lightingState || this.detailTime!==time){
                    const detail=p.detail;
                    this.bindTarget(target);
                    gl.useProgram(detail.program);
                    this.grid(detail,fluid,false);
                    this.texture(2,gl.TEXTURE_3D,this.noise,detail.uNoise);
                    const rest=Boolean(fluid.restA && fluid.restB);
                    this.texture(4,gl.TEXTURE_2D,rest?fluid.restA:fluid.field,detail.uRestA,this.linear);
                    this.texture(5,gl.TEXTURE_2D,rest?fluid.restB:fluid.field,detail.uRestB,this.linear);
                    gl.uniform1i(detail.uHasRest,rest?1:0);
                    gl.uniform1f(detail.uRestBlend,fluid.restBlend??0);
                    gl.uniform1f(detail.uTime,time);
                    gl.drawArrays(gl.TRIANGLES,0,3);
                    this.detailState=this.lightingState;
                    this.detailTime=time;
                }
            }
            const opticalCache=useCache?this.updateOpticalCache(fluid,options,preset,smoke):null;
            pass?.('surface');
            this.drawSurface(surface,fluid,b,aspect,options,preset,mode);
            pass?.('volume');
            const [floor,scale]=({low:[.55,.60],medium:[.63,.74],high:[.75,.86],ultra:[.84,1.]})[options.quality]??[.63,.74];
            const adaptive=Number.isFinite(options.renderScale)?options.renderScale:1;
            const ratio=Math.max(useCache?scale*.70:floor,Math.min(1,scale*adaptive));
            const dim=value=>Math.min(value,Math.max(8,Math.round(value*ratio/8)*8));
            this.renderWidth=dim(w);
            this.renderHeight=dim(h);
            const volume=this.volumeBuffer(this.renderWidth,this.renderHeight);

            this.bindTarget(volume);
            const variant=useCache?'cachedVolume'+fluid.tiles:'volume';
            const ray=p[variant]??(p[variant]=buildProgram(gl,vertex,volumeFragment(this.optics,useCache?fluid.tiles:0)));
            gl.useProgram(ray.program);
            this.camera(ray,b,aspect);
            this.grid(ray,fluid);
            this.texture(2,gl.TEXTURE_3D,this.noise,ray.uNoise);
            this.texture(3,gl.TEXTURE_2D,surface.tex,ray.uSurface);
            this.texture(6,gl.TEXTURE_2D,this.lights.tex,ray.uLights,this.linear);
            this.texture(7,gl.TEXTURE_2D,smoke.tex,ray.uSmokeLighting,this.linear);
            gl.uniform3iv(ray.uLightDims,smoke.dims);
            gl.uniform1i(ray.uLightTiles,smoke.tiles);
            if(opticalCache){
                this.texture(11,gl.TEXTURE_2D,opticalCache.coefficients.tex,ray.uCoefficients,this.linear);
                this.texture(12,gl.TEXTURE_2D,opticalCache.emission.tex,ray.uEmission,this.linear);
                gl.uniform3iv(ray.uOpticalDims,opticalCache.dims);
                gl.uniform1i(ray.uSharedLighting??null,shared?1:0);
                gl.uniform1i(ray.uStableSampling??null,options.stableSampling!==false?1:0);
            }else if(this.optics){
                this.bindOptics(ray,fluid,options);
                const hasRest=Boolean(fluid.restA && fluid.restB);
                this.texture(4,gl.TEXTURE_2D,fluid.restA??fluid.field,ray.uRestA,this.linear);
                this.texture(5,gl.TEXTURE_2D,fluid.restB??fluid.field,ray.uRestB,this.linear);
                gl.uniform1i(ray.uHasRest??null,hasRest?1:0);
                gl.uniform1f(ray.uRestBlend??null,fluid.restBlend??0);
            }else this.texture(4,gl.TEXTURE_2D,detailTexture,ray.uDetail,this.linear);
            const constrainedMedium=this.optics && mode===0 && options.quality==='medium' &&
                options.stableSampling!==false && adaptive<=.84;
            this.raySamples=constrainedMedium?88:(({low:84,medium:112,high:138,ultra:158})[options.quality]??112);
            if(opticalCache){
                const low=options.quality==='low';
                this.raySamples=adaptive<=.70?(low?44:56):low?56:constrainedMedium?64:80;
            }
            gl.uniform1i(ray.uSteps,this.raySamples);
            gl.uniform1i(ray.uMode,mode);
            gl.uniform1i(ray.uPreset,preset);
            gl.drawArrays(gl.TRIANGLES,0,3);

            this.bindTarget(scene);
            pass?.('resolve');
            gl.useProgram(p.resolve.program);
            this.texture(0,gl.TEXTURE_2D,surface.tex,p.resolve.uSurface);
            this.texture(1,gl.TEXTURE_2D,volume.tex,p.resolve.uVolume);
            gl.drawArrays(gl.TRIANGLES,0,3);
        }else{
            pass?.('surface');
            this.drawSurface(scene,fluid,b,aspect,options,preset,mode);
            pass?.('resolve');
            this.renderWidth=w;this.renderHeight=h;this.raySamples=null;
        }
        this.sparks(fluid,b,aspect,time,preset,mode===0);
        gl.bindVertexArray(this.vao);
        if(mode===0){
            gl.useProgram(p.bloom.program);
            for(let i=1;i<=3;i++){
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
        pass?.(null);
        for(let i=0;i<(this.optics?13:9);i++)gl.bindSampler(i,null);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindVertexArray(null);
    }
    destroy(){
        const gl=this.gl;
        this.clearTargets();
        this.clearOpticalCache();
        for(const p of Object.values(this.programs))gl.deleteProgram(p.program);
        for(const b of this.sparkBuffers)gl.deleteBuffer(b);
        for(const v of this.sparkVaos)gl.deleteVertexArray(v);
        if(this.lights){gl.deleteFramebuffer(this.lights.fb);gl.deleteTexture(this.lights.tex);}
        if(this.detailTarget){
            gl.deleteFramebuffer(this.detailTarget.fb);
            gl.deleteTexture(this.detailTarget.tex);
        }
        if(this.volumeTarget){
            gl.deleteFramebuffer(this.volumeTarget.fb);
            gl.deleteTexture(this.volumeTarget.tex);
        }
        if(this.smokeLights){
            gl.deleteFramebuffer(this.smokeLights.fb);
            gl.deleteTexture(this.smokeLights.tex);
        }
        if(this.opticsLut)gl.deleteTexture(this.opticsLut);
        if(this.noise)gl.deleteTexture(this.noise);
        if(this.linear)gl.deleteSampler(this.linear);
        if(this.vao)gl.deleteVertexArray(this.vao);
        if(this.feedback)gl.deleteTransformFeedback(this.feedback);
    }
}

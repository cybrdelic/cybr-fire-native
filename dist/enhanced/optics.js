import {shadowGeometry} from '../lighting.js';

const tableWidth = 256;
const minimumKelvin = 650;
const maximumKelvin = 2650;

// Smooth CIE 1931 2° color matching functions (Wyman, Sloan and Shirley).
// Planck radiance is weighted by the approximate 1/lambda absorption of soot.
function gaussian(wavelength,center,left,right) {
    const x=(wavelength-center)*(wavelength<center?left:right);
    return Math.exp(-.5*x*x);
}

function spectrumSamples() {
    const samples=[];
    for(let nm=380;nm<=780;nm+=5) {
        const x=1.056*gaussian(nm,599.8,.0264,.0323)
              + .362*gaussian(nm,442,.0624,.0374)
              - .065*gaussian(nm,501.1,.049,.0382);
        const y=.821*gaussian(nm,568.8,.0213,.0247)
              + .286*gaussian(nm,530.9,.0613,.0322);
        const z=1.217*gaussian(nm,437,.0845,.0278)
              + .681*gaussian(nm,459,.0385,.0725);
        const weight=Math.pow(560/nm,6);
        samples.push({exponent:1.4387769e7/nm,
            red:weight*(3.2404542*x-1.5371385*y-.4985314*z),
            green:weight*(-.969266*x+1.8760108*y+.041556*z),
            blue:weight*(.0556434*x-.2040259*y+1.0572252*z)});
    }
    return samples;
}

function radianceAt(samples,kelvin) {
    const rgb=[0,0,0];
    for(const band of samples) {
        const occupied=1/Math.expm1(band.exponent/kelvin);
        rgb[0]+=band.red*occupied;
        rgb[1]+=band.green*occupied;
        rgb[2]+=band.blue*occupied;
    }
    return rgb;
}

export function createThermalTable() {
    const samples=spectrumSamples();
    const reference=Math.max(...radianceAt(samples,1800));
    const table=new Float32Array(tableWidth*4);
    for(let i=0;i<tableWidth;i++) {
        const kelvin=minimumKelvin+(maximumKelvin-minimumKelvin)*i/(tableWidth-1);
        const rgb=radianceAt(samples,kelvin);
        const peak=Math.max(...rgb);
        for(let channel=0;channel<3;channel++)table[4*i+channel]=Math.max(0,rgb[channel]/peak);
        table[4*i+3]=peak/reference;
    }
    return table;
}

export function createOpticsLut(gl) {
    const texture=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,tableWidth,1,0,gl.RGBA,gl.FLOAT,createThermalTable());
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    return texture;
}

const smokeOptics=`
float sootSigma(float soot) {
    float c=max(0.,soot);
    return 5.35*c;
}
`;

export const gasOptics = smokeOptics + `
uniform highp sampler2D uOptics,uReaction;
uniform bool uSharedLighting,uHasReaction;

vec4 thermalSpectrum(float kelvin) {
    float t=clamp((kelvin-650.)/2000.,0.,1.);
    return textureLod(uOptics,vec2((.5+255.*t)/256.,.5),0.);
}
vec3 thermalColor(float kelvin) {return thermalSpectrum(kelvin).rgb;}

float resolvedBurn(vec4 state,vec3 p,int preset) {
    if(uHasReaction)return max(0.,sampleGrid(uReaction,p).r);
    return max(0.,state.b*state.a)*(preset==2?8.4:6.8)*smoothstep(.09,.44,state.r);
}

// A calibrated temperature mapping converts the dimensionless solver field.
// The table contains linear RGB chromaticity and temperature-dependent power.
vec3 gasEmission(vec4 state,float thermal,float shell,float sheet,int preset,float burnRate) {
    float soot=max(0.,state.g);
    float mixing=burnRate/(.26+burnRate);
    float participation=smoothstep(.06,.64,mixing)*smoothstep(.24,.83,state.r);
    // Freshly reacting soot near the thin front reaches incandescence.
    // Transported hot soot behind the front contributes a weaker continuum.
    float front=smoothstep(.025,.75,sheet);
    float frontWeight=.44+.56*front;
    float effectiveT=980.+1115.*clamp(thermal,0.,1.22)+240.*participation*front;
    vec4 spectrum=thermalSpectrum(effectiveT);
    float hotSoot=soot/(.085+soot);
    float continuum=.37*(.18+.82*mixing)*smoothstep(.56,1.12,state.r);
    vec3 thermalLight=spectrum.rgb*(spectrum.a*hotSoot*(shell*(.39+2.43*frontWeight)+continuum));
    if(preset==4 || preset==5){
        // Keep texture in the high-radiance interior during the short pulse.
        // Compress only the rendering emissivity, which leaves the solved
        // gas field and the timing of the event untouched.
        float luminance=dot(thermalLight,vec3(.2126,.7152,.0722));
        float shoulder=preset==4?.016:.024;
        thermalLight/=1.+shoulder*luminance;
    }
    float reaction=burnRate/(.35+burnRate);
    float chem=reaction*smoothstep(.035,.43,state.a)*exp(-8.*soot)*(.22+.78*sheet);
    float blueGain=preset==2 || preset==7?2.35:(preset==1?.12:.055);
    return thermalLight+vec3(.035,.30,1.)*(chem*blueGain);
}

vec3 smokeInscatter(vec4 cache,float phaseWeight) {
    // Empty neighbors store zero, occupied cells store 1 + sky transmittance.
    float sky=clamp(cache.a-1.,0.,1.);
    vec3 ambient=uSharedLighting?vec3(.059,.067,.078)*mix(.22,1.,sky):vec3(.043,.051,.063)*mix(.40,1.,sky);
    return ambient+cache.rgb*phaseWeight;
}
`;

export const irradianceBody = gasOptics + `
in vec2 uv;
out vec4 frag;
uniform int uPreset;

float radialEdge(float i) {float t=i/7.;return .96*t*(.39+.61*t);}
float heightEdge(float i) {float t=i/9.;return 2.32*t*(.31+.69*t);}

void main() {
    int sector=int(gl_FragCoord.x);
    vec2 quadrant=vec2((sector&1)==0?-1.:1.,(sector&2)==0?-1.:1.);
    vec3 rgb=vec3(0.),moment=vec3(0.);
    float weight=0.;
    for(int j=0;j<9;j++){
        float ya=heightEdge(float(j)),yb=heightEdge(float(j+1));
        for(int k=0;k<7;k++){
            float za=radialEdge(float(k)),zb=radialEdge(float(k+1));
            for(int i=0;i<7;i++){
                float xa=radialEdge(float(i)),xb=radialEdge(float(i+1));
                vec3 p=.5*vec3(quadrant.x*(xa+xb),ya+yb,quadrant.y*(za+zb));
                vec4 s=sampleGrid(uField,p);
                float heat=max(s.r,0.),t2=heat*heat;
                if(heat<.08)continue;
                vec3 e;
                if(uSharedLighting){
                    float mixture=s.b/(.018+.24*max(0.,s.a)+s.b);
                    float coord=.66*heat+.72*mixture-.70;
                    // This probe covers unresolved folds. Average their emission
                    // over a finite band with the same area as the sharp sheet.
                    float sheet=.67*exp2(-41.*coord*coord);
                    float warm=smoothstep(.17,.50,heat);
                    float shell=warm*(1.-.73*smoothstep(.36,.68,mixture))*(.068+2.10*sheet);
                    float thermal=clamp(.91*heat+.085*sheet,0.,1.07);
                    e=gasEmission(s,thermal,shell,sheet,uPreset,resolvedBurn(s,p,uPreset));
                }else{
                    e=vec3(1.,.341,.083)*t2*t2*(.48+1.75*min(s.g,.85))*smoothstep(.12,.38,heat);
                }
                float dv=(xb-xa)*(yb-ya)*(zb-za);
                rgb+=e*dv;
                float power=max(e.r,max(e.g,e.b))*dv;
                moment+=p*power;weight+=power;
            }
        }
    }
    // Convert integrated emissivity to probe irradiance at the scene exposure.
    float gain=uSharedLighting?5.5:14.;
    vec3 center=weight>.000001?moment/weight:vec3(.14*quadrant.x,.26,.14*quadrant.y);
    frag=int(gl_FragCoord.y)==0?vec4(center,gain*weight):vec4(gain*rgb,1.);
}
`;

export const surfaceLighting = gasOptics + `
uniform highp sampler2D uLights;
` + shadowGeometry + `
struct SfIrradiance {vec3 direct;vec3 fill;vec3 specular;vec3 gloss;float flux;};

float surfaceTransmission(vec3 pos,vec3 ray,float rr) {
    // Inspect the receiver/emitter segment in the same soot field as the camera.
    // Far below the fractional intensity cutoff the irradiance is already weak.
    if(!uSharedLighting || rr>4.)return 1.;
    vec3 q1=pos+ray*.25,q2=pos+ray*.75;
    float a=sootSigma(sampleGrid(uField,q1).g);
    float b=sootSigma(sampleGrid(uField,q2).g);
    return mix(exp(-.5*sqrt(rr)*(a+b)),1.,smoothstep(2.,4.,rr));
}

SfIrradiance sfIncident(vec3 p,vec3 n,vec3 view,bool wood,float logId) {
    SfIrradiance light=SfIrradiance(vec3(0),vec3(0),vec3(0),vec3(0),0.);
    vec3 pos=p+.003*n;
    for(int i=0;i<4;i++){
        vec4 sampleLight=texelFetch(uLights,ivec2(i,0),0);
        light.flux+=sampleLight.w;
        if(sampleLight.w<.00001)continue;
        vec3 ray=sampleLight.xyz-pos;
        float rr=max(.00001,dot(ray,ray));
        vec3 direction=ray*inversesqrt(rr);
        float incidence=max(0.,dot(n,direction));
        float visible=wood && incidence>0.?sfWoodShadow(pos,ray,rr,logId):1.;
        vec3 e=texelFetch(uLights,ivec2(i,1),0).rgb*surfaceTransmission(pos,ray,rr)/(.107+rr);
        if(uMode!=0)e*=.27;
        float reflectivity=max(0.,dot(reflect(-direction,n),view));
        light.fill+=e;
        light.direct+=e*(incidence*visible);
        light.specular+=e*(pow(reflectivity,23.)*visible);
        light.gloss+=e*(pow(reflectivity,5.)*visible);
    }
    return light;
}
`;

export const smokeLightingBody = smokeOptics + `
out vec4 frag;
uniform highp sampler2D uLights;
uniform ivec3 uLightDims;
uniform int uLightTiles,uLightProbes;
uniform bool uLightShading,uSharedLighting;

float sampleSigma(vec3 p) {
    float carbon=max(0.,sampleGrid(uField,p).g);
    if(uSharedLighting)return sootSigma(carbon);
    carbon*=.86;
    return 4.8*carbon*(.46+.54*sqrt(carbon+.02));
}
bool occupiedCell(ivec3 cell) {
    ivec3 start=max(ivec3(0),cell*4-1);
    ivec3 stop=min(uDims-1,cell*4+4);
    for(int z=start.z;z<=stop.z;z++){
        ivec2 tile=ivec2(z%uTiles,z/uTiles)*uDims.xy;
        for(int y=start.y;y<=stop.y;y++){
            for(int x=start.x;x<=stop.x;x++){
                vec2 rg=texelFetch(uField,tile+ivec2(x,y),0).rg;
                if(rg.r+rg.g>=.0045)return true;
            }
        }
    }
    return false;
}
void main() {
    ivec2 px=ivec2(gl_FragCoord.xy);
    ivec3 cell=ivec3(px%uLightDims.xy,px.x/uLightDims.x+(px.y/uLightDims.y)*uLightTiles);
    if(cell.z>=uLightDims.z || !occupiedCell(cell)){frag=vec4(0.);return;}
    if(!uLightShading){frag=vec4(0.,0.,0.,2.);return;}
    vec3 p=(vec3(cell)+.5)/vec3(uLightDims)*gridScale-gridOffset;
    float height=2.4-p.y;
    float tauSky=0.;
    for(int j=0;j<4;j++)tauSky+=sampleSigma(p+vec3(0.,height*(float(j)+.5)*.25,0.));
    float sky=exp(-.25*height*tauSky);
    if(uSharedLighting){
        // Cooler overcast fill comes from a wide slanted hemisphere. A
        // second broad direction keeps the lit folds and lee distinct.
        vec3 side=normalize(vec3(-.52,.73,-.42))*min(1.40,height/.73);
        float tauSide=.5*length(side)*(sampleSigma(p+side*.24)+sampleSigma(p+side*.72));
        sky=mix(sky,exp(-tauSide),.44);
    }
    vec3 incoming=vec3(0.);
    if(uLightProbes==1){
        vec3 total=vec3(0.),moment=vec3(0.);float power=0.;
        for(int k=0;k<4;k++){
            vec4 probe=texelFetch(uLights,ivec2(k,0),0);
            total+=texelFetch(uLights,ivec2(k,1),0).rgb;
            moment+=probe.xyz*probe.w;power+=probe.w;
        }
        if(power>.00001){
            vec3 ray=moment/power-p;
            float d2=dot(ray,ray);
            float tau=.5*sqrt(d2)*(sampleSigma(p+.25*ray)+sampleSigma(p+.75*ray));
            incoming=total*exp(-tau)/(.17+d2);
        }
    }else{
        for(int k=0;k<4;k++){
            vec4 probe=texelFetch(uLights,ivec2(k,0),0);
            if(probe.w<.00001)continue;
            vec3 color=texelFetch(uLights,ivec2(k,1),0).rgb;
            vec3 ray=probe.xyz-p;
            float d2=dot(ray,ray);
            float tau=0.;
            for(int j=0;j<4;j++)tau+=sampleSigma(p+ray*((float(j)+.5)*.25));
            incoming+=color*exp(-.25*sqrt(d2)*tau)/(.17+d2);
        }
    }
    // The soot attenuating the view is predominantly absorbing; only
    // the scattered fraction returns firelight into the camera ray.
    float scatterFraction=uSharedLighting?.023:.176;
    frag=vec4(incoming*scatterFraction,1.+sky);
}
`;

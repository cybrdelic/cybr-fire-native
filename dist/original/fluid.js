import {presetCodes} from '../presets.js';

// A staggered finite-volume velocity grid and a cell-centered reacting scalar grid.
// Each XY slice occupies a tile in a filterable 2D floating-point framebuffer.
const LEVELS = {
  low:    [32, 48, 32, 8, 5, 3],
  medium: [48, 72, 48, 8, 7, 5],
  high:   [64, 96, 64, 8, 9, 6],
  ultra:  [80, 120, 80, 10, 11, 8],
};
const VERT = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0., 1.);
}`;
const preamble = (grid,other,extended) => `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
${grid ? `#define ENHANCED_FIRE
const ivec3 dims=ivec3(${grid.dims.join(',')}), otherDims=ivec3(${other.dims.join(',')});
const int tiles=${grid.tiles}, otherTiles=${other.tiles};` : `uniform ivec3 dims, otherDims;
uniform int tiles, otherTiles;`}
${extended ? '#define EXTENDED_PRESETS\n' : ''}
uniform int preset;
uniform float dt, simTime, fuel, wind, turbulence, oxygen;
uniform vec3 brush;
uniform sampler2D src, aux, extra, obstacle;
#ifdef ENHANCED_FIRE
uniform sampler2D predictedSource, departureCells;
#endif
layout(location=0) out vec4 frag;
const vec3 WORLD = vec3(2., 2.4, 2.);
ivec3 voxel() {
  ivec2 f = ivec2(gl_FragCoord.xy);
  return ivec3(f.x % dims.x, f.y % dims.y, f.x / dims.x + (f.y / dims.y) * tiles);
}
ivec2 loc(ivec3 c, ivec3 d, int t) {
  c = clamp(c, ivec3(0), d - 1);
  return ivec2((c.z % t)*d.x + c.x, (c.z / t)*d.y + c.y);
}
vec4 at(sampler2D s, ivec3 c) { return texelFetch(s, loc(c,dims,tiles), 0); }
vec4 tri(sampler2D s, vec3 g, ivec3 d, int t) {
  g = clamp(g,vec3(0),vec3(d-1));
  int z = int(floor(g.z)), z2 = min(z + 1, d.z - 1);
  vec2 size = vec2(textureSize(s,0));
  vec2 uv = g.xy + .5;
  vec2 off0 = vec2((z % t)*d.x, (z / t)*d.y);
  vec2 off1 = vec2((z2 % t)*d.x, (z2 / t)*d.y);
  return mix(texture(s,(uv+off0)/size),texture(s,(uv+off1)/size),fract(g.z));
}
vec4 interp(sampler2D s, vec3 g) { return tri(s,g,dims,tiles); }
vec3 centeredVelocity(sampler2D s, vec3 g) {
  return vec3(interp(s,g-vec3(.5,0,0)).x,interp(s,g-vec3(0,.5,0)).y,interp(s,g-vec3(0,0,.5)).z);
}
vec3 departure(sampler2D v, vec3 g, float interval) {
  vec3 scale = vec3(dims)/WORLD;
#ifdef ENHANCED_FIRE
  // The first RK2 sample is exactly at a cell center. Recover all three
  // MAC components with four point fetches, including clamped boundary faces.
  ivec3 cell=ivec3(g);
  vec3 a=.5*(at(v,cell).xyz+vec3(at(v,cell-ivec3(1,0,0)).x,
    at(v,cell-ivec3(0,1,0)).y,at(v,cell-ivec3(0,0,1)).z));
#else
  vec3 a = centeredVelocity(v,g);
#endif
  vec3 b = centeredVelocity(v,g - .5*interval*scale*a);
  return g - interval*scale*b;
}
vec3 pos(vec3 g) {return ((g+.5)/vec3(dims))*WORLD - vec3(1.,0.,1.);}
float saturate(float v) {return clamp(v,0.,1.);}
float h31(vec3 k) {
  k = fract(k*.1031);
  k += dot(k,k.yzx + 33.33);
  return fract((k.x+k.y)*k.z);
}
float noise3(vec3 p) {
  vec3 i = floor(p), f = fract(p);
  f = f*f*(3.-2.*f);
  return mix(mix(mix(h31(i),h31(i+vec3(1,0,0)),f.x),
                 mix(h31(i+vec3(0,1,0)),h31(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(h31(i+vec3(0,0,1)),h31(i+vec3(1,0,1)),f.x),
                 mix(h31(i+vec3(0,1,1)),h31(i+vec3(1,1,1)),f.x),f.y),f.z);
}
float atBrush(vec3 p) {
  if (brush.x < 0.) return 0.;
  vec3 d = (p+vec3(1,0,1)) - brush*WORLD;
  return exp(-dot(d,d)*66.);
}
float disk(vec2 p, vec2 r) {
  vec2 a = p/r;
  return exp(-dot(a,a)*1.65);
}
#ifdef EXTENDED_PRESETS
float square(float x){return x*x;}
float burstProgress(float t) {
  return preset==4?smoothstep(.105,.315,t):smoothstep(.11,.39,t);
}
float burstRate(float t) {
  return (burstProgress(t+dt)-burstProgress(t))/max(dt,.0001);
}
vec3 burstCenter(float t) {
  return preset==4?vec3(-.045,.324+.22*t,0.):vec3(-.21+.73*t,.12,.045);
}
float burstMask(vec3 p,float t) {
  vec3 delta=p-burstCenter(t);
  if(any(greaterThan(abs(delta),preset==4?vec3(.72):vec3(.95,.34,.68))))return 0.;
  // The uneven source seeds instability on the expanding boundary. After
  // this short impulse, the velocity field alone moves the reacting gas.
  float lump=noise3(p*9.2+vec3(4.7,2.18,3.1));
  if(preset==4){
    vec3 direction=delta/max(length(delta),.03);
    float lobes=noise3(direction*2.9+vec3(9.1,12.7,23.));
    float radius=.19+.42*t+.19*(lobes-.5)+.113*(lump-.5);
    float r=length(delta*vec3(.88,1.12,1.05));
    float shell=exp(-square((r-radius)/.103));
    return clamp(shell*(.54+.58*lump)+.14*(1.-smoothstep(.04,radius,r)),0.,1.);
  }
  float r=length(delta*vec3(.69,3.05,1.2));
  float radius=.31+.52*t+.095*(lump-.5);
  float band=exp(-square((r-.66*radius)/.185));
  return band*(.60+.56*lump)*smoothstep(.015,.085,p.y);
}
float ventPulse(float t) {
  float cycle=fract(.76*t+.934);
  return smoothstep(.015,.19,cycle)*(1.-smoothstep(.36,.565,cycle));
}
#endif
float sourceMask(vec3 p, float t) {
#ifdef EXTENDED_PRESETS
  if(preset<6){
    float rate=burstRate(t);
    return rate>0.?rate*burstMask(p,t+.5*dt):0.;
  }
#endif
  vec2 u = p.xz;
  float base=at(obstacle,voxel()).w;
  float flicker=.88+.12*sin(t*8.+u.x*11.)*cos(t*5.1+u.y*14.);
#ifdef EXTENDED_PRESETS
  if(preset==7)return base*flicker*ventPulse(t+.5*dt);
  float licks=smoothstep(.35,1.,sin(t*1.31+u.x*17.+u.y*9.)*
      cos(t*3.1-u.x*13.+u.y*18.));
  return base*(.089+.70*licks)*flicker;
#else
  return base*flicker;
#endif
}
vec3 faces(float encoded) {
  int mask=int(encoded);
  return vec3(mask&1,(mask>>1)&1,(mask>>2)&1);
}
vec3 impermeable(vec3 v, ivec3 c) {
  return v*faces(at(obstacle,c).x);
}
`;

const seed = `
vec4 seeded(vec3 p) {
#ifdef EXTENDED_PRESETS
  if(preset==6){
    float bed=disk(p.xz,vec2(.47,.31))*exp(-square((p.y-.076)/.055));
    return vec4(bed*.24*sqrt(fuel*oxygen),bed*.071,bed*.12*fuel,oxygen*(1.-.19*bed));
  }
  return vec4(0.,0.,0.,oxygen);
#else
  float y = max(0.,p.y-.1);
  float gain = sqrt(fuel*oxygen);
  float lean = wind*y*.54;
  vec2 a=p.xz-vec2(lean,0.);
  vec3 n = vec3(a.x*3.2,y*3.85,a.y*3.2);
  a += (.025+.095*y)*vec2(noise3(n*1.7+8.)*2.-1.,noise3(n*1.9+34.)*2.-1.);
  a -= .095*y*vec2(sin(y*7.+.8),sin(y*5.-.4));
  float height = (preset==2 ? 1.77 : preset==1 ? 1.36 : 1.47)*(.43+.75*gain);
  float r = max(.035,1.-y/(height*1.34));
  float m;
  if(preset==2) m=disk(a,vec2(.11+.045*y));
  else if(preset==1) m=disk(a,vec2(.41*r+.05));
  else {
    m=max(max(disk(a-vec2(-.19,-.085)*r,vec2(.19,.14)*r+.07),
              disk(a-vec2(.16,.10)*r,vec2(.20,.15)*r+.055)),
              disk(a-vec2(.06,-.16)*r,vec2(.14,.12)*r+.03));
  }
  float tongues = noise3(vec3(p.x*6.,p.y*2.2,p.z*6.)+9.);
  float h = height*(.79+.54*tongues);
  float top = 1.-smoothstep(h*.51,h*1.1,y);
  float heat = pow(m,.47)*top*(.88+.18*tongues)*gain*1.23;
  float smoke = disk(a,vec2(.23+.16*y)) * smoothstep(.15,.85,y)*(.14+.50*tongues)*gain;
  float f = m*(1.-smoothstep(.06,.78,y))*.42*fuel;
  return vec4(heat,smoke,f,clamp(oxygen-m*top*.62,0.,1.));
#endif
}
`;
const programs = {
  obstacles: `
  // The same tapered polygonal prisms and biased end planes used by the ray caster.
  float logDist(vec3 v,vec3 a,vec3 b,vec2 radii,float seed) {
    vec3 axis=normalize(b-a),side=normalize(cross(axis,vec3(0,1,0))),up=cross(side,axis);
    v-=a;float len=length(b-a);
    vec3 p=vec3(dot(v,side)/radii.x,dot(v,up)/radii.y,dot(v,axis));
    float ez=p.z+.016*sin(seed)*p.x, dist=max(-ez,ez-len);
    for(int j=0;j<8;++j){
      float angle=float(j)*.7853982+seed*.71;
      vec2 n=vec2(cos(angle),sin(angle));
      float plane=.89+.065*sin(float(j)*3.17+seed*5.)-(j==0?.26:0.);
      dist=max(dist,(dot(n,p.xy)+p.z*.09/len-plane)/length(n/radii));
    }
    return dist;
  }
  float sdf(vec3 p){
    if(p.y>.40 || max(abs(p.x),abs(p.z))>.60) return 1.;
#ifdef EXTENDED_PRESETS
    if(preset!=7)return 1.;
    float r=.125,rad=length(p.xz);
    return min(max(rad-r*.91,max(.002-p.y,p.y-.051)),max(rad-r,max(.049-p.y,p.y-.072)));
#else
    if(preset==1 || preset==2) {
      float r=preset==2?.138:.405,rad=length(p.xz);
      return min(max(rad-r*.91,max(.002-p.y,p.y-.052)),max(rad-r,max(.050-p.y,p.y-.074)));
    }
    float d=logDist(p,vec3(-.39,.070,-.11),vec3(.30,.066,-.17),vec2(.080,.068),1.2);
    d=min(d,logDist(p,vec3(-.32,.068,.18),vec3(.39,.070,.085),vec2(.075,.067),2.7));
    d=min(d,logDist(p,vec3(-.28,.076,-.31),vec3(.17,.087,.30),vec2(.079,.064),3.4));
    d=min(d,logDist(p,vec3(-.20,.127,.30),vec3(.19,.176,-.32),vec2(.082,.068),4.1));
    d=min(d,logDist(p,vec3(.38,.065,.24),vec3(-.22,.202,-.19),vec2(.052,.048),5.2));
    d=min(d,logDist(p,vec3(-.37,.045,.10),vec3(.24,.185,-.08),vec2(.042,.040),6.9));
    return d;
#endif
  }
  void main(){
    ivec3 c=voxel();vec3 p=pos(vec3(c));
    float d=sdf(p);if(d<=0.){frag=vec4(0);return;}
    int positive=0,negative=0;
    if(c.x<dims.x-1 && sdf(pos(vec3(c+ivec3(1,0,0))))>0.) positive|=1;
    if(sdf(pos(vec3(c+ivec3(0,1,0))))>0.) positive|=2;
    if(c.z<dims.z-1 && sdf(pos(vec3(c+ivec3(0,0,1))))>0.) positive|=4;
    if(c.x>0 && sdf(pos(vec3(c-ivec3(1,0,0))))>0.) negative|=1;
    if(c.y>0 && sdf(pos(vec3(c-ivec3(0,1,0))))>0.) negative|=2;
    if(c.z>0 && sdf(pos(vec3(c-ivec3(0,0,1))))>0.) negative|=4;
    float source;
#ifdef EXTENDED_PRESETS
    source=0.;
    if(preset==6){
      source=disk(p.xz,vec2(.56,.34))*exp(-square((p.y-.082)/.054));
      source*=.47+.82*noise3(vec3(p.x*19.,8.5,p.z*15.));
    }else if(preset==7){
      source=disk(p.xz,vec2(.136))*exp(-square((p.y-.108)/.075))*step(.072,p.y);
    }
#else
    if(preset==1 || preset==2){
      float lateral=preset==2 ? disk(p.xz,vec2(.145)) : 1.-smoothstep(.235,.405,length(p.xz));
      source=lateral*exp(-pow((p.y-.098)/(preset==2?.072:.064),2.))*step(.074,p.y);
    }else{
      float core=1.-smoothstep(.15,.48,length(p.xz));
      source=exp(-pow(d/.064,2.))*(.35+.77*core)*smoothstep(.008,.09,p.y);
    }
#endif
    frag=vec4(float(positive),float(negative),1.,source);
  }`,
  seedField: seed+`void main() {frag=at(obstacle,voxel()).z*seeded(pos(vec3(voxel())));}`,
  seedVelocity: seed+`void main() {
    ivec3 c=voxel(); vec3 p=pos(vec3(c));
#ifdef EXTENDED_PRESETS
    frag=vec4(impermeable(vec3(0.),c),0.);
#else
    float t=seeded(p).x;
    vec3 v=vec3(wind*(.16+.74*t)+.19*t*sin(9.*p.y+3.*p.z),t*(1.55+.93*float(preset==2)),.16*t*cos(8.*p.y+5.*p.x));
    frag=vec4(impermeable(v,c),0.);
#endif
  }`,
  velocity: `void main(){
    ivec3 c=voxel(); vec3 g=vec3(c), p=pos(g), h=vec3(dims)/WORLD;
    vec3 v=interp(src,departure(src,g,dt)).xyz*exp(-.30*dt);
    vec4 s=at(aux,c);
    vec3 w=at(extra,c).xyz;
    vec3 grad=.5*h*vec3(length(at(extra,c+ivec3(1,0,0)).xyz)-length(at(extra,c-ivec3(1,0,0)).xyz),
      length(at(extra,c+ivec3(0,1,0)).xyz)-length(at(extra,c-ivec3(0,1,0)).xyz),
      length(at(extra,c+ivec3(0,0,1)).xyz)-length(at(extra,c-ivec3(0,0,1)).xyz));
    vec3 confinement = cross(grad/max(length(grad),.0001),w)*(.55+1.6*turbulence)/h.x;
    float m = sourceMask(p,simTime);
    float t = simTime;
    vec3 stir = vec3(sin(8.7*p.y+5.*p.z-5.2*t)+.53*sin(17.*p.z+3.2*t),
                  .38*sin(7.*p.x+6.*p.z+3.8*t),cos(7.3*p.y-5.*p.x-4.7*t)+.47*cos(12.*p.x-3.5*t));
#ifdef EXTENDED_PRESETS
    vec3 acc=confinement + stir*(s.x+min(m,1.))*(.52+2.8*turbulence);
    if(preset<6){
      float isFireball=float(preset==4);
      if(m>0.){
        vec3 fromCenter=p-burstCenter(simTime+.5*dt);
        if(preset==5)fromCenter=vec3(fromCenter.x+.22,fromCenter.y*.43,fromCenter.z);
        vec3 direction=fromCenter/max(length(fromCenter),.08);
        float rise=0.7+1.1*isFireball;
        acc+=m*(direction*(3.9-.65*isFireball)+vec3(.58*(1.-isFireball),rise,0.));
      }
      acc.y+=(3.25+.6*isFireball)*s.x-.31*s.y;
      acc.x+=((wind+.10*(1.-isFireball))*1.23-v.x)*(.65+.40*s.x);
    }else{
      acc.y+=(preset==6?2.1:4.05)*s.x-.27*s.y + m*(preset==7?14.3:1.2)*fuel;
      acc.x += (wind*1.28-v.x)*(.72+.42*s.x);
    }
#else
    vec3 acc=confinement + stir*(s.x+m)*(.52+2.8*turbulence);
    acc.y += 4.05*s.x-.24*s.y + m*(preset==2 ? 12. : 1.8)*fuel;
    acc.x += (wind*1.28-v.x)*(.72+.42*s.x);
#endif
    acc.y += 4.8*atBrush(p);
    v=clamp(v+dt*acc,vec3(-4.),vec3(4.));
    frag=vec4(impermeable(v,c),0.);
  }`,
  derivatives: `uniform float expansion;
  void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;
    vec3 v=at(src,c).xyz;
    vec3 l=at(src,c-ivec3(1,0,0)).xyz,r=at(src,c+ivec3(1,0,0)).xyz;
    vec3 b=at(src,c-ivec3(0,1,0)).xyz,t=at(src,c+ivec3(0,1,0)).xyz;
    vec3 a=at(src,c-ivec3(0,0,1)).xyz,f=at(src,c+ivec3(0,0,1)).xyz;
    float div=dot(k,v-vec3(c.x==0?0.:l.x,c.y==0?0.:b.y,c.z==0?0.:a.z));
    vec3 curl=.5*vec3((t.z-b.z)*k.y-(f.y-a.y)*k.z,(f.x-a.x)*k.z-(r.z-l.z)*k.x,
                     (r.y-l.y)*k.x-(t.x-b.x)*k.y);
#ifdef ENHANCED_FIRE
    // Project onto divergence S from the preceding chemistry step (first-order splitting).
    div -= expansion*at(aux,c).b;
#endif
    frag=vec4(curl,div);
  }`,
  restrict: `uniform float expansion;
  void main() {
    ivec3 c=voxel();vec3 g=2.*vec3(c)+.5,p=pos(vec3(c));
    float div;
    // The two coarser presets have connected coarse gas domains in all scenes.
    if(dims.x<32 || p.y>.42 || max(abs(p.x),abs(p.z))>.62) {
      div=tri(src,g,otherDims,otherTiles).w;
    } else {
      // A coarse gas pocket can have a different topology from its eight
      // children. Restrict MAC flux on its open faces so pressure remains
      // compatible with the coarse Neumann boundary, including sealed pockets.
      vec4 cell=at(obstacle,c);
      vec3 positive=vec3(tri(aux,g+vec3(.5,0,0),otherDims,otherTiles).x,
                         tri(aux,g+vec3(0,.5,0),otherDims,otherTiles).y,
                         tri(aux,g+vec3(0,0,.5),otherDims,otherTiles).z);
      vec3 negative=vec3(tri(aux,g-vec3(1.5,0,0),otherDims,otherTiles).x,
                         tri(aux,g-vec3(0,1.5,0),otherDims,otherTiles).y,
                         tri(aux,g-vec3(0,0,1.5),otherDims,otherTiles).z);
      div=dot(vec3(dims)/WORLD,faces(cell.x)*positive-faces(cell.y)*negative);
#ifdef ENHANCED_FIRE
      div-=expansion*tri(extra,g,otherDims,otherTiles).b;
#endif
    }
    frag=vec4(0.,0.,0.,div);
  }`,
  prolongate: `void main() {
    vec3 g=.5*(vec3(voxel())+.5)-.5;
    frag=vec4(tri(src,g,otherDims,otherTiles).x,0.,0.,0.);
  }`,
  pressure: `void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;k*=k;
    float l=at(src,c-ivec3(1,0,0)).x,r=at(src,c+ivec3(1,0,0)).x;
    float b=at(src,c-ivec3(0,1,0)).x,t=c.y==dims.y-1 ? 0. : at(src,c+ivec3(0,1,0)).x;
    float a=at(src,c-ivec3(0,0,1)).x,f=at(src,c+ivec3(0,0,1)).x;
    vec4 cell=at(obstacle,c);
    vec3 plus=faces(cell.x),minus=faces(cell.y);
    float diagonal=dot(k,plus+minus);
    // Coarsening may seal a one-cell air pocket even when fine-scale flux crosses
    // its eight children. An isolated pressure unknown has no gradient to solve.
    float pr=diagonal>0. ? (dot(k,minus*vec3(l,b,a)+plus*vec3(r,t,f))-at(aux,c).w)/diagonal : 0.;
    frag=vec4(cell.z*pr,0,0,0);
  }`,
  project: `void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;
    float p=at(aux,c).x;
    vec3 adj=vec3(at(aux,c+ivec3(1,0,0)).x,c.y==dims.y-1 ? 0. : at(aux,c+ivec3(0,1,0)).x,at(aux,c+ivec3(0,0,1)).x);
    frag=vec4(impermeable(at(src,c).xyz-k*(adj-p),c),0.);
  }`,
  seedRest: `layout(location=1) out vec4 secondRest;
  void main() {
    vec3 identity=(vec3(voxel())+.5)/vec3(dims);
    frag=vec4(identity,1.);
    secondRest=frag;
  }`,
  transportRest: `layout(location=1) out vec4 secondRest;
#ifdef ENHANCED_FIRE
  layout(location=2) out vec4 predictedGas;
  layout(location=3) out float departureIndex;
#endif
  uniform ivec2 restartRest;
  void main() {
    vec3 g=vec3(voxel()), identity=(g+.5)/vec3(dims);
    vec3 back=departure(extra,g,dt);
#ifdef ENHANCED_FIRE
    // The material and reacting scalars follow the same characteristic.
    // Encode the limiter's lower corner in R32F (exact through 2^24).
    ivec3 base=clamp(ivec3(floor(back)),ivec3(-1),dims-1)+1;
    departureIndex=float(base.x+(dims.x+1)*(base.y+(dims.y+1)*base.z));
    predictedGas=at(obstacle,voxel()).z*interp(predictedSource,back);
#endif
    vec3 p=pos(g);
    // Freshly emitted fluid and inflow boundaries carry undeformed material coordinates.
    float feed=sourceMask(p,simTime)*fuel+atBrush(p);
    float influx=1.-smoothstep(.015,.065,min(min(identity.x,1.-identity.x),min(identity.z,1.-identity.z)));
    float relax=1.-exp(-dt*(6.5*feed+5.*influx));
    vec3 a=restartRest.x!=0 ? identity : mix(interp(src,back).xyz,identity,relax);
    vec3 b=restartRest.y!=0 ? identity : mix(interp(aux,back).xyz,identity,relax);
    frag=vec4(a,1.);
    secondRest=vec4(b,1.);
  }`,
  transport: `void main() {
    vec3 g=vec3(voxel());
    frag=at(obstacle,voxel()).z*interp(src,departure(aux,g,dt));
  }`,
  chemistry: `
#ifdef ENHANCED_FIRE
  layout(location=1) out vec4 reaction;
#endif
  void main() {
    ivec3 c=voxel(); vec3 g=vec3(c),p=pos(g);
#ifdef ENHANCED_FIRE
    reaction=vec4(0.);
#endif
    if(at(obstacle,c).z<.5){frag=vec4(0);return;}
    vec4 s=at(aux,c);
    // MacCormack correction with a local extrema limiter at the original departure.
    vec4 reverse=interp(aux,departure(extra,g,-dt));
    vec4 corrected=s+.5*(at(src,c)-reverse);
#ifdef ENHANCED_FIRE
    int index=int(texelFetch(departureCells,ivec2(gl_FragCoord.xy),0).r);
    int yz=index/(dims.x+1);
    ivec3 i=ivec3(index%(dims.x+1),yz%(dims.y+1),yz/(dims.y+1))-1;
#else
    ivec3 i=ivec3(floor(departure(extra,g,dt)));
#endif
    vec4 lo=at(src,i), hi=lo;
    for (int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
      vec4 a=at(src,i+ivec3(x,y,z));lo=min(lo,a);hi=max(hi,a);
    }
    s=clamp(corrected,lo,hi);
    float m=sourceMask(p,simTime), paint=atBrush(p);
    float edge=smoothstep(.79,.995,max(abs(p.x),abs(p.z)));
    s.w=mix(s.w,oxygen,1.-exp(-dt*(.32+edge*4.)));
    s.w=min(s.w,oxygen);
#ifdef EXTENDED_PRESETS
    float feed=(m*(preset==4?2.9:preset==5?2.54:preset==6?4.25:10.1)+paint*5.2)*fuel;
    float pilot=((preset==4?4.55:preset==5?4.2:preset==6?6.65:9.2)*m+6.3*paint)*sqrt(fuel*oxygen);
#else
    float feed=(m*(preset==2?9.:7.)+paint*5.2)*fuel;
    float pilot=(7.7*m+6.3*paint)*sqrt(fuel*oxygen);
#endif
#ifdef ENHANCED_FIRE
    float fuelSource=feed*(1.-s.z);
#endif
    s.z += dt*feed*(1.-s.z);
#ifdef ENHANCED_FIRE
    float pilotSource=pilot*max(0.,1.13-s.x);
#endif
    s.x += dt*pilot*max(0.,1.13-s.x);
    float ign = smoothstep(.09,.44,s.x);
#ifdef EXTENDED_PRESETS
    float burn=min(min(s.z,s.w/.77),dt*(preset==7?8.4:6.8)*s.z*ign*s.w);
#else
    float burn=min(min(s.z,s.w/.77),dt*(preset==2?8.4:6.8)*s.z*ign*s.w);
#endif
    s.z=max(0.,s.z-burn)*exp(-dt*(.20+edge*2.));
    s.w=max(0.,s.w-.77*burn);
#ifdef ENHANCED_FIRE
    // Constant-pressure ideal-gas dilatation from chemical heating: D(T)/Dt / T.
    // Temperature is normalized; .32 is ambient on that scale. The resolved
    // fraction controls a subcell expansion closure for this visual model.
    // Limit peak source strength near thin sheets with poorly resolved reactants.
    float rate=burn/dt;
    const float resolvedFraction=.22;
    float dilatation=min(1.25,resolvedFraction*2.43*rate/(.32+s.x+2.43*burn));
#ifdef EXTENDED_PRESETS
    if(preset<6)dilatation=min(5.8,dilatation+2.8*m);
#endif
    reaction=vec4(rate,fuelSource,dilatation,pilotSource);
#endif
    s.x+=burn*2.43;
#ifdef EXTENDED_PRESETS
    float cooling=preset==4?1.59:preset==5?1.83:preset==6?1.23:1.10;
    s.x*=exp(-dt*(cooling+.58*s.x+edge*3.5));
    float rich=1.-smoothstep(.06,.78,s.w);
    float sootYield=preset==4?1.12:preset==5?1.32:preset==6?1.38:.24;
    s.y+=sootYield*((.17+.68*rich)*burn + dt*.26*s.z*rich*ign);
    if(preset==5 || preset==6)s.y+=dt*(preset==5?.13:.22)*m*rich;
    s.y*=exp(-dt*(.23+edge*3.+.64*s.w*ign));
#else
    s.x*=exp(-dt*(.85+.54*s.x+edge*3.5));
    float rich=1.-smoothstep(.06,.78,s.w);
    s.y+=(.17+.68*rich)*burn + dt*.26*s.z*rich*ign;
    s.y*=exp(-dt*(.23+edge*3.+.64*s.w*ign));
#endif
    frag=clamp(s,vec4(0.),vec4(1.32,1.,1.,1.));
  }`,
};
const extendedProgramNames = ['obstacles','seedField','seedVelocity','velocity','transportRest','chemistry'];

function makeProgram(gl, vertex, label, body, grid, other, extended=false) {
  const shader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(shader,preamble(grid,other,extended)+body);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) {
    const info=gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error('Fluid '+label+' shader: '+info);
  }
  const handle=gl.createProgram();
  gl.attachShader(handle,vertex); gl.attachShader(handle,shader); gl.linkProgram(handle);
  gl.deleteShader(shader);
  if (!gl.getProgramParameter(handle,gl.LINK_STATUS)) {
    const info=gl.getProgramInfoLog(handle);gl.deleteProgram(handle);throw new Error('Fluid '+label+': '+info);
  }
  const loc={};
  for (const key of ['dims','tiles','otherDims','otherTiles','preset','dt','simTime','fuel','oxygen','wind','turbulence','brush','src','aux','extra','restartRest','obstacle','expansion','predictedSource','departureCells']) {
    loc[key]=gl.getUniformLocation(handle,key);
  }
  gl.useProgram(handle);
  ['src','aux','extra'].forEach((name,index)=>gl.uniform1i(loc[name],index));
  gl.uniform1i(loc.obstacle,3);
  gl.uniform1i(loc.predictedSource,4);
  gl.uniform1i(loc.departureCells,5);
  return {handle,loc};
}

export class Fluid {
  constructor(gl,quality='medium',enhanced=false) {
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('Floating-point render targets are unavailable.');
    this.gl=gl;
    this.enhanced=enhanced;
    this.extendedPrograms=null;
    const level = LEVELS[quality] || LEVELS.medium;
    this.dims=level.slice(0,3); this.tiles=level[3];
    this.coarseIterations=level[4]; this.fineIterations=level[5];
    this.pressureIterations=this.coarseIterations+this.fineIterations;
    this.fine={dims:this.dims,tiles:this.tiles};
    this.coarse={dims:this.dims.map(n=>n/2),tiles:Math.min(8,this.tiles)};
    this.targets=[]; this.programs={}; this.vao=gl.createVertexArray();
    const vertex=gl.createShader(gl.VERTEX_SHADER);
    try {
      gl.shaderSource(vertex,VERT); gl.compileShader(vertex);
      if (!gl.getShaderParameter(vertex,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(vertex));
      for(const [label,body] of Object.entries(programs)){
        if(enhanced && label==='transport')continue;
        const grid=label==='restrict'?this.coarse:this.fine;
        const other=label==='restrict'?this.fine:this.coarse;
        this.programs[label]=makeProgram(gl,vertex,label,body,enhanced?grid:null,other);
        if(enhanced && (label==='obstacles'||label==='pressure')){
          this.programs[label+'Coarse']=makeProgram(gl,vertex,label,body,this.coarse,this.fine);
        }
      }
      this.s=Array.from({length:3},()=>this.makeTarget(this.fine));
      this.v=Array.from({length:2},()=>this.makeTarget(this.fine));
      this.deriv=this.makeTarget(this.fine); this.rhs=this.makeTarget(this.coarse);
      this.pc=Array.from({length:2},()=>this.makeTarget(this.coarse,true));
      this.pf=Array.from({length:2},()=>this.makeTarget(this.fine,true));
      this.solidFine=this.makeTarget(this.fine);this.solidCoarse=this.makeTarget(this.coarse);
      if(enhanced){
        this.rates=this.makeTarget(this.fine);
        // Chemistry ping-pongs s[0] and s[2]. Both FBOs write the same diagnostic
        // atlas. Other passes use their own FBO, so its last value stays available.
        for(const target of [this.s[0],this.s[2]]){
          gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo);
          gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,gl.TEXTURE_2D,this.rates.tex,0);
          gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
          if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('Unable to allocate the reaction atlas.');
        }
      }
      if(enhanced)this.departureCells=this.makeTarget(this.fine,false,gl.R32F);
      this.restPairs=quality==='low' && !enhanced ? null : [this.makeRestPair(),this.makeRestPair()];
      this.reset();
    } catch(error) {this.destroy(); throw error;} finally {gl.deleteShader(vertex);}
  }

  get field() {return this.s[0].tex;}
  get velocity() {return this.v[0].tex;}
  get reaction() {return this.rates?.tex ?? null;}

  get restA() {return this.restPairs?.[this.restRead]?.[0].tex ?? null;}
  get restB() {return this.restPairs?.[this.restRead]?.[1].tex ?? null;}

  makeRestPair() {
    const gl=this.gl;
    const a=this.makeTarget(this.fine), b=this.makeTarget(this.fine);
    gl.bindFramebuffer(gl.FRAMEBUFFER,a.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,gl.TEXTURE_2D,b.tex,0);
    if(this.enhanced){
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT2,gl.TEXTURE_2D,this.s[1].tex,0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT3,gl.TEXTURE_2D,this.departureCells.tex,0);
    }
    gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error('Unable to allocate dual rest volume.');
    return [a,b];
  }

  makeTarget(grid,scalar=false,format=null) {
    const gl=this.gl,[nx,ny,nz]=grid.dims;
    const width=nx*grid.tiles,height=ny*Math.ceil(nz/grid.tiles);
    const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texStorage2D(gl.TEXTURE_2D,1,format??(scalar?gl.R16F:gl.RGBA16F),width,height);
    const filter=format===gl.R32F?gl.NEAREST:gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    const fbo=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
    const target={grid,width,height,tex,fbo}; this.targets.push(target);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error('Unable to allocate fluid volume.');
    return target;
  }

  begin(params={},dt=0,time=0) {
    const gl=this.gl;
    this.settings=params; this.dt=dt; this.time=time;
    const code=presetCodes[params.preset]??presetCodes.wood;
    this.presetCode=this.enhanced || code<4?code:presetCodes.wood;
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST); gl.disable(gl.CULL_FACE);
    gl.colorMask(true,true,true,true);
  }

  pass(name,target,inputs=[],other=null) {
    const useCoarse=this.enhanced && target.grid===this.coarse && (name==='obstacles'||name==='pressure');
    const key=useCoarse?name+'Coarse':name;
    const program=(this.presetCode>=4 && this.extendedPrograms[key])||this.programs[key];
    const gl=this.gl, {handle,loc}=program, p=this.settings;
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo); gl.viewport(0,0,target.width,target.height);
    gl.useProgram(handle);
    gl.uniform3iv(loc.dims,target.grid.dims);gl.uniform1i(loc.tiles,target.grid.tiles);
    if (other) {gl.uniform3iv(loc.otherDims,other.dims);gl.uniform1i(loc.otherTiles,other.tiles);}
    if (loc.expansion!==null)gl.uniform1f(loc.expansion,p.expansion===false?0:1);
    if (loc.fuel!==null) gl.uniform1f(loc.fuel,p.fuel??.72);
    if (loc.wind!==null) gl.uniform1f(loc.wind,p.wind??0);
    if (loc.turbulence!==null) gl.uniform1f(loc.turbulence,p.turbulence??.68);
    if (loc.oxygen!==null) gl.uniform1f(loc.oxygen,p.oxygen??.90);
    if (loc.preset!==null) gl.uniform1i(loc.preset,this.presetCode);
    if (loc.brush!==null) gl.uniform3fv(loc.brush,p.brush||[-1,-1,-1]);
    if (loc.simTime!==null) gl.uniform1f(loc.simTime,this.time);
    if (loc.dt!==null) gl.uniform1f(loc.dt,this.dt);
    if (loc.restartRest!==null) gl.uniform2i(loc.restartRest,this.restartA,this.restartB);
    if(loc.obstacle!==null) {
      const mask=target.grid===this.fine?this.solidFine:this.solidCoarse;
      gl.activeTexture(gl.TEXTURE3);gl.bindSampler(3,null);gl.bindTexture(gl.TEXTURE_2D,mask.tex);
    }
    if(this.enhanced && name==='transportRest'){
      gl.activeTexture(gl.TEXTURE4);gl.bindSampler(4,null);gl.bindTexture(gl.TEXTURE_2D,this.s[0].tex);
    }
    if(this.enhanced && name==='chemistry'){
      gl.activeTexture(gl.TEXTURE5);gl.bindSampler(5,null);gl.bindTexture(gl.TEXTURE_2D,this.departureCells.tex);
    }
    inputs.forEach((input,index)=>{
      gl.activeTexture(gl.TEXTURE0+index);gl.bindSampler(index,null);gl.bindTexture(gl.TEXTURE_2D,input.tex);
    });
    gl.drawArrays(gl.TRIANGLES,0,3);
  }

  prepareExtendedPrograms() {
    if(this.extendedPrograms)return;
    const gl=this.gl, staged={}, vertex=gl.createShader(gl.VERTEX_SHADER);
    try{
      gl.shaderSource(vertex,VERT); gl.compileShader(vertex);
      if(!gl.getShaderParameter(vertex,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(vertex));
      for(const name of extendedProgramNames){
        staged[name]=makeProgram(gl,vertex,name,programs[name],this.fine,this.coarse,true);
      }
      staged.obstaclesCoarse=makeProgram(gl,vertex,'obstaclesCoarse',programs.obstacles,this.coarse,this.fine,true);
      this.extendedPrograms=staged;
    }catch(error){
      for(const p of Object.values(staged))gl.deleteProgram(p.handle);
      throw error;
    }finally{gl.deleteShader(vertex);}
  }

  reset(params={}) {
    const gl=this.gl;
    if(this.enhanced && presetCodes[params.preset]>=4)this.prepareExtendedPrograms();
    this.begin(params);
    for (const target of this.targets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo);
      gl.clearBufferfv(gl.COLOR,0,[0,0,0,0]);
    }
    this.pass('obstacles',this.solidFine);this.pass('obstacles',this.solidCoarse);
    this.restRead=0;this.restTime=0;this.restBlend=1;
    if(this.enhanced){
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.restPairs[0][0].fbo);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
    }
    if(this.restPairs)this.pass('seedRest',this.restPairs[0][0]);
    if(this.enhanced){
      for(const pair of this.restPairs){
        gl.bindFramebuffer(gl.FRAMEBUFFER,pair[0].fbo);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1,gl.COLOR_ATTACHMENT2,gl.COLOR_ATTACHMENT3]);
      }
    }
    if(this.enhanced){
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.s[0].fbo);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    }
    this.pass('seedField',this.s[0]);
    if(this.enhanced)gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
    this.pass('seedVelocity',this.v[0]);
    this.pass('derivatives',this.deriv,this.enhanced?[this.v[0],this.rates]:[this.v[0]]);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindVertexArray(null);
    this.coarseRead=0;
  }

  step(dt,params,time) {
    if (!(dt>0)) return;
    this.begin(params,Math.min(dt,1/15),time);
    if(this.presetCode>=4 && !this.extendedPrograms)this.prepareExtendedPrograms();
    this.pass('velocity',this.v[1],[this.v[0],this.s[0],this.deriv]);
    this.pass('derivatives',this.deriv,this.enhanced?[this.v[1],this.rates]:[this.v[1]]);
    // Nested-grid projection. Preserve coarse pressure as the next warm start.
    this.pass('restrict',this.rhs,this.enhanced?[this.deriv,this.v[1],this.rates]:[this.deriv,this.v[1]],this.fine);
    let coarse=this.coarseRead;
    for(let i=0;i<this.coarseIterations;i++) {
      this.pass('pressure',this.pc[1-coarse],[this.pc[coarse],this.rhs]);coarse=1-coarse;
    }
    this.coarseRead=coarse;
    this.pass('prolongate',this.pf[0],[this.pc[coarse]],this.coarse);
    let fine=0;
    for(let i=0;i<this.fineIterations;i++) {
      this.pass('pressure',this.pf[1-fine],[this.pf[fine],this.deriv]);fine=1-fine;
    }
    this.pass('project',this.v[0],[this.v[1],this.pf[fine]]);
    if(this.restPairs) {
      // Reset each material map only when its visible contribution is near zero.
      const halfCycle=.80, period=2*halfCycle, t0=this.restTime;
      this.restTime+=this.dt;
      this.restartA=Number(Math.floor(this.restTime/period)!==Math.floor(t0/period));
      this.restartB=Number(Math.floor((this.restTime+halfCycle)/period)!==Math.floor((t0+halfCycle)/period));
      this.restBlend=.5+.5*Math.cos(2*Math.PI*this.restTime/period);
      const next=1-this.restRead;
      this.pass('transportRest',this.restPairs[next][0],[...this.restPairs[this.restRead],this.v[0]]);
      this.restRead=next;
    }
    if(!this.enhanced)this.pass('transport',this.s[1],[this.s[0],this.v[0]]);
    this.pass('chemistry',this.s[2],[this.s[0],this.s[1],this.v[0]]);
    [this.s[0],this.s[2]]=[this.s[2],this.s[0]];
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,null);this.gl.bindVertexArray(null);
  }

  destroy() {
    const gl=this.gl;
    for(const target of this.targets){gl.deleteFramebuffer(target.fbo);gl.deleteTexture(target.tex);}
    for(const p of Object.values(this.programs)) gl.deleteProgram(p.handle);
    if(this.extendedPrograms)for(const p of Object.values(this.extendedPrograms))gl.deleteProgram(p.handle);
    gl.deleteVertexArray(this.vao);
  }
}

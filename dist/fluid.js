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
const PRE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
uniform ivec3 dims, otherDims, woodDims;
uniform int tiles, otherTiles, preset;
uniform float dt, simTime, fuel, wind, turbulence, oxygen;
uniform vec3 brush;
uniform bool coldStart;
uniform sampler2D src, aux, extra, obstacle, wood, restOne, restTwo, departures;
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
// The current fragment is already in atlas coordinates. Adjacent XY cells use
// unit pixel offsets; a Z neighbor either advances a tile or wraps to its row.
struct Stencil {
  ivec2 center, left, right, below, above, back, front;
};
Stencil stencil(ivec3 cell) {
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  int tile=cell.z%tiles;
  ivec2 previous=tile==0 ? ivec2((tiles-1)*dims.x,-dims.y) : ivec2(-dims.x,0);
  ivec2 next=tile==tiles-1 ? ivec2((1-tiles)*dims.x,dims.y) : ivec2(dims.x,0);
  return Stencil(pixel,
    pixel-ivec2(int(cell.x>0),0),pixel+ivec2(int(cell.x<dims.x-1),0),
    pixel-ivec2(0,int(cell.y>0)),pixel+ivec2(0,int(cell.y<dims.y-1)),
    cell.z>0 ? pixel+previous : pixel,cell.z<dims.z-1 ? pixel+next : pixel);
}
vec4 point(sampler2D tex,ivec2 pixel) {return texelFetch(tex,pixel,0);}
vec3 cellVelocity(sampler2D tex,Stencil n) {
  return .5*(point(tex,n.center).xyz+vec3(point(tex,n.left).x,point(tex,n.below).y,point(tex,n.back).z));
}

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
vec3 departure(sampler2D v, vec3 g, vec3 a, float interval) {
  vec3 displacement=interval*vec3(dims)/WORLD;
  vec3 b=centeredVelocity(v,g-.5*displacement*a);
  return g-displacement*b;
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
float sourceMask(vec3 p, float t) {
  vec2 u = p.xz;
  float base=at(obstacle,voxel()).w;
  float flicker=.88+.12*sin(t*8.+u.x*11.)*cos(t*5.1+u.y*14.);
  return base*flicker;
}
bool hasWood() {return preset==0 || preset==3;}
const ivec3 AXIS[6]=ivec3[6](ivec3(1,0,0),ivec3(-1,0,0),
  ivec3(0,1,0),ivec3(0,-1,0),ivec3(0,0,1),ivec3(0,0,-1));
vec4 solidAt(ivec3 c) {
  if(any(lessThan(c,ivec3(0))) || any(greaterThanEqual(c,woodDims))) return vec4(0.);
  return texelFetch(wood,loc(c,woodDims,tiles),0);
}
// Sum volatile emission and sensible heat across the faces touching wood.
// The face weighting converts a surface flux to a cell-volume source.
vec2 woodSource(ivec3 c, vec2 faceMasks) {
  // A gas cell surrounded by open faces cannot share a face with wood.
  if(c.y>woodDims.y || all(equal(faceMasks,vec2(7.)))) return vec2(0.);
  vec3 invSpacing=vec3(dims)/WORLD;
  vec2 result=vec2(0.);
  for(int j=0;j<6;++j){
    vec4 s=solidAt(c+AXIS[j]);
    result+=invSpacing[j/2]*vec2(s.a,s.r*s.r)*.042;
  }
  return result;
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
  if(coldStart) return vec4(0.,0.,0.,oxygen);
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
    if(preset==1 || preset==2){
      float lateral=preset==2 ? disk(p.xz,vec2(.145)) : 1.-smoothstep(.235,.405,length(p.xz));
      source=lateral*exp(-pow((p.y-.098)/(preset==2?.072:.064),2.))*step(.074,p.y);
    }else{
      float core=1.-smoothstep(.15,.48,length(p.xz));
      source=exp(-pow(d/.064,2.))*(.35+.77*core)*smoothstep(.008,.09,p.y);
    }
    frag=vec4(float(positive),float(negative),1.,source);
  }`,
  seedWood: `void main() {
    ivec3 c=voxel();vec3 p=((vec3(c)+.5)/vec3(otherDims))*WORLD-vec3(1.,0.,1.);
    if(texelFetch(obstacle,loc(c,otherDims,otherTiles),0).z>.5){frag=vec4(0.);return;}
    float core=1.-smoothstep(.12,.55,length(p.xz));
    float warmth=coldStart?0.:sqrt(fuel*oxygen)*(.46+.53*core);
    float dark=coldStart?0.:.20*core;
    frag=vec4(warmth,1.-dark,dark*.45,0.);
  }`,
  evolveWood: `void main() {
    ivec3 c=voxel();vec3 p=((vec3(c)+.5)/vec3(otherDims))*WORLD-vec3(1.,0.,1.);
    if(texelFetch(obstacle,loc(c,otherDims,otherTiles),0).z>.5){frag=vec4(0.);return;}
    vec4 solid=at(src,c);
    float conducted=0., gasHeat=0., gasAir=0., facesToAir=0.;
    for (int j=0;j<6;++j) {
      ivec3 n=c+AXIS[j];
      if(any(lessThan(n,ivec3(0))) || any(greaterThanEqual(n,otherDims))) continue;
      ivec2 uv=loc(n,otherDims,otherTiles);
      if(texelFetch(obstacle,uv,0).z>.5) {
        vec4 fluid=texelFetch(aux,uv,0);
        gasHeat+=fluid.r;gasAir+=fluid.a;facesToAir+=1.;
      } else {
        conducted+=at(src,n).r-solid.r;
      }
    }
    float invFaces=1./max(facesToAir,1.);
    gasHeat*=invFaces;gasAir*=invFaces;
    float exposed=1.-exp(-.64*facesToAir);
    // Geometry remains fixed. Material fractions follow finite virgin-wood
    // conversion and char oxidation, with separate heat capacities for gas and wood.
    float igniter=6.1*atBrush(p);
    float temp=solid.x;
    float pyro=fuel*solid.y*exposed*.115*smoothstep(.28,.64,temp);
    float vented=pyro*.69;
    float ox=min(oxygen,gasAir*.58+oxygen*.42);
    float charBurn=solid.z*exposed*.24*ox*smoothstep(.26,.66,temp);
    float convection=exposed*(4.3*max(0.,gasHeat-temp)-.68*max(0.,temp-gasHeat));
    float loss=.12*temp+exposed*.41*temp*temp;
    temp+=dt*(3.*conducted+convection+15.2*charBurn-2.5*pyro-loss+igniter);
    float virgin=max(0.,solid.y-dt*pyro);
    float charMass=clamp(solid.z+dt*(.31*pyro-charBurn),0.,1.-virgin);
    frag=vec4(clamp(temp,0.,1.32),virgin,charMass,vented);
  }`,
  seedField: seed+`void main() {frag=at(obstacle,voxel()).z*seeded(pos(vec3(voxel())));}`,
  seedVelocity: seed+`void main() {
    ivec3 c=voxel(); vec3 p=pos(vec3(c));
    float t=seeded(p).x;
    vec3 v=vec3(wind*(.16+.74*t)+.19*t*sin(9.*p.y+3.*p.z),t*(1.55+.93*float(preset==2)),.16*t*cos(8.*p.y+5.*p.x));
    frag=vec4(impermeable(v,c),0.);
  }`,
  velocity: `void main(){
    ivec3 c=voxel(); vec3 g=vec3(c), p=pos(g), h=vec3(dims)/WORLD;
    Stencil n=stencil(c);
    vec4 solid=point(obstacle,n.center);
    if(solid.z<.5){frag=vec4(0.);return;}
    vec3 v=interp(src,departure(src,g,cellVelocity(src,n),dt)).xyz*exp(-.30*dt);
    vec4 s=point(aux,n.center);
    vec3 w=point(extra,n.center).xyz;
    vec3 grad=.5*h*vec3(length(point(extra,n.right).xyz)-length(point(extra,n.left).xyz),
      length(point(extra,n.above).xyz)-length(point(extra,n.below).xyz),
      length(point(extra,n.front).xyz)-length(point(extra,n.back).xyz));
    vec3 confinement = cross(grad/max(length(grad),.0001),w)*(.55+1.6*turbulence)/h.x;
    float m = hasWood()?5.*woodSource(c,solid.xy).x:sourceMask(p,simTime)*fuel;
    float t = simTime;
    vec3 stir = vec3(sin(8.7*p.y+5.*p.z-5.2*t)+.53*sin(17.*p.z+3.2*t),
                  .38*sin(7.*p.x+6.*p.z+3.8*t),cos(7.3*p.y-5.*p.x-4.7*t)+.47*cos(12.*p.x-3.5*t));
    vec3 acc=confinement + stir*(s.x+m)*(.52+2.8*turbulence);
    acc.y += 4.05*s.x-.24*s.y + m*(preset==2 ? 12. : 1.8);
    acc.x += (wind*1.28-v.x)*(.72+.42*s.x);
    acc.y += 4.8*atBrush(p);
    v=clamp(v+dt*acc,vec3(-4.),vec3(4.));
    frag=vec4(v*faces(solid.x),0.);
  }`,
  derivatives: `void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;
    Stencil n=stencil(c);
    vec3 v=point(src,n.center).xyz;
    vec3 l=point(src,n.left).xyz,r=point(src,n.right).xyz;
    vec3 b=point(src,n.below).xyz,t=point(src,n.above).xyz;
    vec3 a=point(src,n.back).xyz,f=point(src,n.front).xyz;
    float div=dot(k,v-vec3(c.x==0?0.:l.x,c.y==0?0.:b.y,c.z==0?0.:a.z));
    vec3 curl=.5*vec3((t.z-b.z)*k.y-(f.y-a.y)*k.z,(f.x-a.x)*k.z-(r.z-l.z)*k.x,
                     (r.y-l.y)*k.x-(t.x-b.x)*k.y);
    frag=vec4(curl,div);
  }`,
  restrict: `void main() {
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
    }
    frag=vec4(0.,0.,0.,div);
  }`,
  prolongate: `void main() {
    vec3 g=.5*(vec3(voxel())+.5)-.5;
    frag=vec4(tri(src,g,otherDims,otherTiles).x,0.,0.,0.);
  }`,
  pressure: `void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;k*=k;
    Stencil n=stencil(c);
    vec4 cell=point(obstacle,n.center);
    if(cell.z<.5){frag=vec4(0.);return;}
    float l=point(src,n.left).x,r=point(src,n.right).x;
    float b=point(src,n.below).x,t=c.y==dims.y-1 ? 0. : point(src,n.above).x;
    float a=point(src,n.back).x,f=point(src,n.front).x;
    vec3 plus=faces(cell.x),minus=faces(cell.y);
    float diagonal=dot(k,plus+minus);
    // A sealed one-cell pressure unknown has no gradient to solve.
    float pr=diagonal>0. ? (dot(k,minus*vec3(l,b,a)+plus*vec3(r,t,f))-point(aux,n.center).w)/diagonal : 0.;
    frag=vec4(pr,0,0,0);
  }`,
  project: `void main() {
    ivec3 c=voxel(); vec3 k=vec3(dims)/WORLD;
    Stencil n=stencil(c);
    float center=point(aux,n.center).x;
    vec3 adj=vec3(point(aux,n.right).x,c.y==dims.y-1 ? 0. : point(aux,n.above).x,point(aux,n.front).x);
    frag=vec4((point(src,n.center).xyz-k*(adj-center))*faces(point(obstacle,n.center).x),0.);
  }`,
  seedRest: `layout(location=1) out vec4 secondRest;
  void main() {
    vec3 identity=(vec3(voxel())+.5)/vec3(dims);
    frag=vec4(identity,1.);
    secondRest=frag;
  }`,
  transport: `
  layout(location=1) out vec4 departureCell;
  void main() {
    ivec3 c=voxel(); Stencil n=stencil(c);
    if(point(obstacle,n.center).z<.5){frag=departureCell=vec4(0.);return;}
    vec3 g=vec3(c);
    vec3 back=departure(aux,g,cellVelocity(aux,n),dt);
    frag=interp(src,back);
    departureCell=vec4(floor(back),0.);
  }`,
  transportRest: `
  layout(location=1) out vec4 departureCell;
  layout(location=2) out vec4 firstRest;
  layout(location=3) out vec4 secondRest;
  uniform ivec2 restartRest;
  void main() {
    ivec3 c=voxel();
    Stencil n=stencil(c);
    vec3 g=vec3(c), identity=(g+.5)/vec3(dims);
    vec4 solid=point(obstacle,n.center);
    vec3 back=departure(aux,g,cellVelocity(aux,n),dt);
    frag=solid.z*interp(src,back);
    departureCell=vec4(floor(back),0.);
    vec3 p=pos(g);
    // Newly fed fluid and lateral boundaries carry undeformed material coordinates.
    float feed=(hasWood()?woodSource(c,solid.xy).x*7.:sourceMask(p,simTime)*fuel)+atBrush(p);
    float influx=1.-smoothstep(.015,.065,min(min(identity.x,1.-identity.x),min(identity.z,1.-identity.z)));
    float relax=1.-exp(-dt*(6.5*feed+5.*influx));
    vec3 a=restartRest.x!=0 ? identity : mix(interp(restOne,back).xyz,identity,relax);
    vec3 b=restartRest.y!=0 ? identity : mix(interp(restTwo,back).xyz,identity,relax);
    firstRest=vec4(a,1.);
    secondRest=vec4(b,1.);
  }`,
  chemistry: `void main() {
    ivec3 c=voxel(); vec3 g=vec3(c),p=pos(g);
    Stencil n=stencil(c);
    vec4 solidMask=point(obstacle,n.center);
    if(solidMask.z<.5){frag=vec4(0);return;}
    vec4 s=point(aux,n.center);
    // MacCormack correction. Read the exact integer forward sample cell saved
    // by advection instead of tracing the same characteristic a second time.
    vec4 reverse=interp(aux,departure(extra,g,cellVelocity(extra,n),-dt));
    vec4 corrected=s+.5*(point(src,n.center)-reverse);
    ivec3 i=ivec3(point(departures,n.center).xyz);
    ivec3 loCell=clamp(i,ivec3(0),dims-1),hiCell=clamp(i+1,ivec3(0),dims-1);
    ivec2 tileA=ivec2(loCell.z%tiles,loCell.z/tiles)*dims.xy;
    ivec2 tileB=ivec2(hiCell.z%tiles,hiCell.z/tiles)*dims.xy;
    vec4 a0=point(src,tileA+loCell.xy), a1=point(src,tileA+ivec2(hiCell.x,loCell.y));
    vec4 a2=point(src,tileA+ivec2(loCell.x,hiCell.y)), a3=point(src,tileA+hiCell.xy);
    vec4 b0=point(src,tileB+loCell.xy), b1=point(src,tileB+ivec2(hiCell.x,loCell.y));
    vec4 b2=point(src,tileB+ivec2(loCell.x,hiCell.y)), b3=point(src,tileB+hiCell.xy);
    vec4 lo=min(min(min(a0,a1),min(a2,a3)),min(min(b0,b1),min(b2,b3)));
    vec4 hi=max(max(max(a0,a1),max(a2,a3)),max(max(b0,b1),max(b2,b3)));
    s=clamp(corrected,lo,hi);
    float m=sourceMask(p,simTime), paint=atBrush(p);
    float edge=smoothstep(.79,.995,max(abs(p.x),abs(p.z)));
    s.w=mix(s.w,oxygen,1.-exp(-dt*(.32+edge*4.)));
    s.w=min(s.w,oxygen);
    vec2 solid=hasWood()?woodSource(c,solidMask.xy):vec2(0.);
    float feed=hasWood()?solid.x*79.:m*(preset==2?9.:7.)*fuel;
    s.z += dt*feed*(1.-s.z);
    float pilot=hasWood() || coldStart?0.:7.7*m*sqrt(fuel*oxygen);
    s.x += dt*(pilot+8.5*paint)*max(0.,1.17-s.x);
    if(hasWood()) s.x+=dt*solid.y*.95*max(0.,1.-s.x);
    float ign = smoothstep(.09,.44,s.x);
    float burn=min(min(s.z,s.w/.77),dt*(preset==2?8.4:6.8)*s.z*ign*s.w);
    s.z=max(0.,s.z-burn)*exp(-dt*(.20+edge*2.));
    s.w=max(0.,s.w-.77*burn);
    s.x+=burn*2.43;
    s.x*=exp(-dt*(.85+.54*s.x+edge*3.5));
    float rich=1.-smoothstep(.06,.78,s.w);
    s.y+=(.17+.68*rich)*burn + dt*.26*s.z*rich*ign;
    s.y*=exp(-dt*(.23+edge*3.+.64*s.w*ign));
    frag=clamp(s,vec4(0.),vec4(1.32,1.,1.,1.));
  }`,
};

function makeProgram(gl, vertex, label, body) {
  const shader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(shader,PRE+body);
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
  for (const key of ['dims','tiles','otherDims','otherTiles','preset','dt','simTime','fuel','oxygen','wind','turbulence','brush','src','aux','extra','restartRest','obstacle','wood','woodDims','coldStart','restOne','restTwo','departures']) {
    loc[key]=gl.getUniformLocation(handle,key);
  }
  gl.useProgram(handle);
  ['src','aux','extra'].forEach((name,index)=>gl.uniform1i(loc[name],index));
  gl.uniform1i(loc.obstacle,3);
  gl.uniform1i(loc.wood,4);
  gl.uniform1i(loc.restOne,5);
  gl.uniform1i(loc.restTwo,6);
  gl.uniform1i(loc.departures,7);
  return {handle,loc};
}

export class Fluid {
  constructor(gl,quality='medium') {
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('Floating-point render targets are unavailable.');
    this.gl=gl;
    const level = LEVELS[quality] || LEVELS.medium;
    this.dims=level.slice(0,3); this.tiles=level[3];
    this.coarseIterations=level[4]; this.fineIterations=level[5];
    this.pressureIterations=this.coarseIterations+this.fineIterations;
    this.fine={dims:this.dims,tiles:this.tiles};
    this.coarse={dims:this.dims.map(n=>n/2),tiles:Math.min(8,this.tiles)};
    this.woodDims=[this.dims[0],Math.ceil(this.dims[1]*.40/2.4),this.dims[2]];
    this.woodGrid={dims:this.woodDims,tiles:this.tiles};
    this.targets=[]; this.advectionFramebuffers=[]; this.programs={}; this.vao=gl.createVertexArray();
    const vertex=gl.createShader(gl.VERTEX_SHADER);
    try {
      gl.shaderSource(vertex,VERT); gl.compileShader(vertex);
      if (!gl.getShaderParameter(vertex,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(vertex));
      for(const [label,body] of Object.entries(programs)) this.programs[label]=makeProgram(gl,vertex,label,body);
      this.s=Array.from({length:3},()=>this.makeTarget(this.fine));
      this.v=Array.from({length:2},()=>this.makeTarget(this.fine));
      this.deriv=this.makeTarget(this.fine); this.rhs=this.makeTarget(this.coarse);
      this.pc=Array.from({length:2},()=>this.makeTarget(this.coarse,true));
      this.pf=Array.from({length:2},()=>this.makeTarget(this.fine,true));
      this.solidFine=this.makeTarget(this.fine);this.solidCoarse=this.makeTarget(this.coarse);
      this.woodPair=Array.from({length:2},()=>this.makeTarget(this.woodGrid));
      this.restPairs=quality==='low' ? null : [this.makeRestPair(),this.makeRestPair()];
      this.departures=this.makeTarget(this.fine);
      // One pass writes predicted gas, its limiter cell and both optional
      // material maps. Static attachments let the two rest FBOs ping-pong.
      this.advectionTargets=this.restPairs
        ? this.restPairs.map(pair=>this.makeAdvectionTarget(pair))
        : [this.makeAdvectionTarget([])];

      this.reset();
    } catch(error) {this.destroy(); throw error;} finally {gl.deleteShader(vertex);}
  }

  get field() {return this.s[0].tex;}
  get velocity() {return this.v[0].tex;}
  // RGBA16F at gas-grid spacing, cropped to the lower 0.40 world units.
  // Channels are solid heat, virgin fraction, char fraction and volatile flux.
  get wood() {return this.woodScene?this.woodPair[this.woodRead].tex:null;}

  get restA() {return this.restPairs?.[this.restRead]?.[0].tex ?? null;}
  get restB() {return this.restPairs?.[this.restRead]?.[1].tex ?? null;}

  makeRestPair() {
    const gl=this.gl;
    const a=this.makeTarget(this.fine), b=this.makeTarget(this.fine);
    gl.bindFramebuffer(gl.FRAMEBUFFER,a.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT1,gl.TEXTURE_2D,b.tex,0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error('Unable to allocate dual rest volume.');
    return [a,b];
  }

  makeAdvectionTarget(restPair) {
    const gl=this.gl, fbo=gl.createFramebuffer();
    this.advectionFramebuffers.push(fbo);
    gl.bindFramebuffer(gl.FRAMEBUFFER,fbo);
    const attachments=[gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1];
    if(restPair.length)attachments.push(gl.COLOR_ATTACHMENT2,gl.COLOR_ATTACHMENT3);
    [this.s[1],this.departures,...restPair].forEach((target,index)=>{
      gl.framebufferTexture2D(gl.FRAMEBUFFER,attachments[index],gl.TEXTURE_2D,target.tex,0);
    });
    gl.drawBuffers(attachments);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) throw new Error('Unable to allocate coupled advection target.');
    return {...this.s[1],fbo};
  }

  makeTarget(grid,scalar=false) {
    const gl=this.gl,[nx,ny,nz]=grid.dims;
    const width=nx*grid.tiles,height=ny*Math.ceil(nz/grid.tiles);
    const tex=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texStorage2D(gl.TEXTURE_2D,1,scalar?gl.R16F:gl.RGBA16F,width,height);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
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
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST); gl.disable(gl.CULL_FACE);
    gl.colorMask(true,true,true,true);
  }

  pass(name,target,inputs=[],other=null) {
    const gl=this.gl, {handle,loc}=this.programs[name], p=this.settings;
    gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo); gl.viewport(0,0,target.width,target.height);
    gl.useProgram(handle);
    gl.uniform3iv(loc.dims,target.grid.dims);gl.uniform1i(loc.tiles,target.grid.tiles);
    if (other) {gl.uniform3iv(loc.otherDims,other.dims);gl.uniform1i(loc.otherTiles,other.tiles);}
    if (loc.fuel!==null) gl.uniform1f(loc.fuel,p.fuel??.72);
    if (loc.wind!==null) gl.uniform1f(loc.wind,p.wind??0);
    if (loc.turbulence!==null) gl.uniform1f(loc.turbulence,p.turbulence??.68);
    if (loc.oxygen!==null) gl.uniform1f(loc.oxygen,p.oxygen??.90);
    if (loc.preset!==null) gl.uniform1i(loc.preset,({wood:0,pool:1,jet:2,crosswind:3})[p.preset]??0);
    if (loc.brush!==null) gl.uniform3fv(loc.brush,p.brush||[-1,-1,-1]);
    if (loc.simTime!==null) gl.uniform1f(loc.simTime,this.time);
    if (loc.dt!==null) gl.uniform1f(loc.dt,this.dt);
    if (loc.restartRest!==null) gl.uniform2i(loc.restartRest,this.restartA,this.restartB);
    if (loc.coldStart!==null) gl.uniform1i(loc.coldStart,p.coldStart?1:0);
    if(loc.obstacle!==null) {
      const mask=target.grid===this.coarse?this.solidCoarse:this.solidFine;
      gl.activeTexture(gl.TEXTURE3);gl.bindSampler(3,null);gl.bindTexture(gl.TEXTURE_2D,mask.tex);
    }
    if (loc.wood!==null) {
      gl.uniform3iv(loc.woodDims,this.woodDims);
      gl.activeTexture(gl.TEXTURE4);gl.bindSampler(4,null);gl.bindTexture(gl.TEXTURE_2D,this.woodPair[this.woodRead].tex);
    }
    if(loc.restOne!==null && this.restPairs) {
      const pair=this.restPairs[this.restRead];
      gl.activeTexture(gl.TEXTURE5);gl.bindSampler(5,null);gl.bindTexture(gl.TEXTURE_2D,pair[0].tex);
      gl.activeTexture(gl.TEXTURE6);gl.bindSampler(6,null);gl.bindTexture(gl.TEXTURE_2D,pair[1].tex);
    }
    if(loc.departures!==null) {
      gl.activeTexture(gl.TEXTURE7);gl.bindSampler(7,null);gl.bindTexture(gl.TEXTURE_2D,this.departures.tex);
    }
    inputs.forEach((input,index)=>{
      gl.activeTexture(gl.TEXTURE0+index);gl.bindSampler(index,null);gl.bindTexture(gl.TEXTURE_2D,input.tex);
    });
    gl.drawArrays(gl.TRIANGLES,0,3);
  }

  reset(params={}) {
    const gl=this.gl;
    this.begin(params);
    this.woodRead=0;
    this.restRead=0;this.restTime=0;this.restBlend=1;
    this.woodScene=params.preset!=='pool' && params.preset!=='jet';
    for (const target of this.targets) {
      gl.bindFramebuffer(gl.FRAMEBUFFER,target.fbo);
      gl.clearBufferfv(gl.COLOR,0,[0,0,0,0]);
    }
    this.pass('obstacles',this.solidFine);this.pass('obstacles',this.solidCoarse);
    if(this.woodScene)this.pass('seedWood',this.woodPair[0],[],this.fine);
    if(this.restPairs) this.pass('seedRest',this.restPairs[0][0]);
    this.pass('seedField',this.s[0]);this.pass('seedVelocity',this.v[0]);
    this.pass('derivatives',this.deriv,[this.v[0]]);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindVertexArray(null);
    this.coarseRead=0;
  }

  step(dt,params,time) {
    if (!(dt>0)) return;
    this.begin(params,Math.min(dt,1/15),time);
    if(this.woodScene) {
      const next=1-this.woodRead;
      this.pass('evolveWood',this.woodPair[next],[this.woodPair[this.woodRead],this.s[0]],this.fine);
      this.woodRead=next;
    }
    this.pass('velocity',this.v[1],[this.v[0],this.s[0],this.deriv]);
    this.pass('derivatives',this.deriv,[this.v[1]]);
    // Nested-grid projection. Preserve coarse pressure as the next warm start.
    this.pass('restrict',this.rhs,[this.deriv,this.v[1]],this.fine);
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
      this.pass('transportRest',this.advectionTargets[next],[this.s[0],this.v[0]]);
      this.restRead=next;
    } else {
      this.pass('transport',this.advectionTargets[0],[this.s[0],this.v[0]]);
    }
    this.pass('chemistry',this.s[2],[this.s[0],this.s[1],this.v[0]]);
    [this.s[0],this.s[2]]=[this.s[2],this.s[0]];
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,null);this.gl.bindVertexArray(null);
  }

  destroy() {
    const gl=this.gl;
    for(const target of this.targets){gl.deleteFramebuffer(target.fbo);gl.deleteTexture(target.tex);}
    for(const framebuffer of this.advectionFramebuffers)gl.deleteFramebuffer(framebuffer);
    for(const p of Object.values(this.programs)) gl.deleteProgram(p.handle);
    gl.deleteVertexArray(this.vao);
  }
}

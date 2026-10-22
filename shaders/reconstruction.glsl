// V8: reconstruct material state before nonlinear spectral classification.
// Mode 0 is the original preclassified trilinear optical cache (ablation).
// Mode 1 is trilinear state + Planck classification.
// Mode 2 is a tensor product of monotonic C1 Hermite state reconstructions.
layout(binding=2) uniform sampler3D physicalState;
layout(binding=3) uniform sampler3D radianceTable;
uniform int reconstructionMode;
uniform float physicalCapacity;
uniform float physicalExtinction;
vec4 stateTexel(ivec3 p){return texelFetch(physicalState,clamp(p,ivec3(0),dims-1),0);}
vec4 limitedSlope(vec4 a,vec4 b){
 vec4 m=.5*(a+b);
 return sign(m)*min(abs(m),2.*min(abs(a),abs(b)))*vec4(greaterThan(a*b,vec4(0.)));
}
vec4 hermiteBounded(vec4 a,vec4 b,vec4 c,vec4 d,float t){
 vec4 m0=limitedSlope(b-a,c-b),m1=limitedSlope(c-b,d-c);
 float t2=t*t,t3=t2*t;
 return clamp((2.*t3-3.*t2+1.)*b+(t3-2.*t2+t)*m0+(-2.*t3+3.*t2)*c+(t3-t2)*m1,min(b,c),max(b,c));
}
vec4 reconstructedState(vec3 uv){
 if(reconstructionMode<2)return textureLod(physicalState,uv,0.);
 vec3 x=uv*vec3(dims)-.5;ivec3 b=ivec3(floor(x));vec3 f=fract(x);
 vec4 zs[4];
 for(int k=0;k<4;k++){
  vec4 ys[4];
  for(int j=0;j<4;j++){
   ivec3 c=b+ivec3(-1,j-1,k-1);
   ys[j]=hermiteBounded(stateTexel(c),stateTexel(c+ivec3(1,0,0)),stateTexel(c+ivec3(2,0,0)),stateTexel(c+ivec3(3,0,0)),f.x);
  }
  zs[k]=hermiteBounded(ys[0],ys[1],ys[2],ys[3],f.y);
 }
 return hermiteBounded(zs[0],zs[1],zs[2],zs[3],f.z);
}
vec4 physicalOptics(vec3 uv){
 vec4 q=reconstructedState(uv);
 float T=300.+max(q.z,0.)/physicalCapacity;
 float t=clamp((T-300.)/5700.,0.,1.);
 vec3 B=textureLod(radianceTable,vec3((t*511.+.5)/512.,.5,.5),0.).rgb;
 float extinction=max(q.w,0.)*physicalExtinction;
 return vec4(B*extinction*(1.-scatterAlbedo),extinction);
}
vec4 mediumSample(vec3 uv){return reconstructionMode==0?textureLod(opticalTexture,uv,0.):physicalOptics(uv);}

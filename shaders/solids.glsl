// Shared world-space geometry in meters.
layout(std430,binding=4) readonly buffer SOLIDS {vec4 solidData[];};
uniform int solidCount;
float sdfOne(vec3 p,int i){
 vec3 c=solidData[4*i].xyz;int kind=int(solidData[4*i].w);vec4 shape=solidData[4*i+1];
 if(kind==0)return length(p-c)-shape.w;
 if(kind==1){vec3 q=abs(p-c)-shape.xyz;return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.);}
 vec3 ba=solidData[4*i+2].xyz-c,pa=p-c;return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.,1.))-shape.w;
}
float solidDistance(vec3 p){float d=1.e20;for(int i=0;i<solidCount;i++)d=min(d,sdfOne(p,i));return d;}
vec3 solidNormal(vec3 p,int i){float e=1.e-4;return normalize(vec3(sdfOne(p+vec3(e,0,0),i)-sdfOne(p-vec3(e,0,0),i),sdfOne(p+vec3(0,e,0),i)-sdfOne(p-vec3(0,e,0),i),sdfOne(p+vec3(0,0,e),i)-sdfOne(p-vec3(0,0,e),i)));}
float sphereHit(vec3 ro,vec3 rd,vec3 c,float r){vec3 oc=ro-c;float b=dot(oc,rd),disc=b*b-dot(oc,oc)+r*r;if(disc<0.)return 1.e20;float t=-b-sqrt(disc);return t>=0.?t:1.e20;}
float oneSolidHit(vec3 ro,vec3 rd,int i){
 vec3 a=solidData[4*i].xyz;vec4 s=solidData[4*i+1];int kind=int(solidData[4*i].w);
 if(kind==0)return sphereHit(ro,rd,a,s.w);
 if(kind==1){vec3 lo=a-s.xyz,hi=a+s.xyz;float entry=-1.e20,exit=1.e20;
  for(int k=0;k<3;k++){if(abs(rd[k])<1.e-10){if(ro[k]<lo[k]||ro[k]>hi[k])return 1.e20;}
  else{float u=(lo[k]-ro[k])/rd[k],v=(hi[k]-ro[k])/rd[k];entry=max(entry,min(u,v));exit=min(exit,max(u,v));}}
  return exit>=max(entry,0.)&&entry>=0.?entry:1.e20;
 }
 vec3 b=solidData[4*i+2].xyz,ba=b-a,oa=ro-a;
 float bb=dot(ba,ba),br=dot(ba,rd),bo=dot(ba,oa),rr=dot(rd,oa),oo=dot(oa,oa);
 float A=bb-br*br,B=bb*rr-bo*br,C=bb*oo-bo*bo-s.w*s.w*bb,h=B*B-A*C,t=1.e20;
 if(h>=0.&&A>1.e-10){float u=(-B-sqrt(h))/A,y=bo+u*br;if(u>=0.&&y>0.&&y<bb)t=u;}
 return min(t,min(sphereHit(ro,rd,a,s.w),sphereHit(ro,rd,b,s.w)));
}
float firstSolidHit(vec3 ro,vec3 rd,out int which){float best=1.e20;which=-1;for(int i=0;i<solidCount;i++){float t=oneSolidHit(ro,rd,i);if(t<best){best=t;which=i;}}return best;}
vec3 clipSolidSegment(vec3 from,vec3 to){
 if(solidCount==0)return to;vec3 delta=to-from;float distance=length(delta);if(distance<1.e-8)return to;
 int obj;float t=firstSolidHit(from,delta/distance,obj);return t<distance?from+delta/distance*max(0.,t-1.e-4):to;
}

// Conservative render acceleration, NOT sparse simulation storage.
layout(std430,binding=1) readonly buffer BB {float brickBounds[];};
uniform ivec3 brickDims;
uniform int brickSize;
uniform int emptySpaceSkipping;
uniform float transmittanceCutoff;
vec2 volumeBoxHit(vec3 ro,vec3 rd,vec3 size){
 vec3 inv=1./rd,a=-ro*inv,b=(size-ro)*inv;
 vec3 mn=min(a,b),mx=max(a,b);
 return vec2(max(max(mn.x,mn.y),mn.z),min(min(mx.x,mx.y),mx.z));
}
int skipEmptySamples(vec3 p,vec3 rd,float entry,float sampleT,float ds,float fraction,int index,int count,out int spanEnd){
 spanEnd=count;
 if(emptySpaceSkipping==0)return index;
 float side=float(brickSize)*spacing;
 ivec3 b=clamp(ivec3(floor(p/side)),ivec3(0),brickDims-1);
 bool empty=brickBounds[index3(b,brickDims)]==0.;
 vec3 plane=(vec3(b)+step(vec3(0.),rd))*side;
 vec3 travel=(plane-p)/rd;
 float distance=1.e20;
 for(int axis=0;axis<3;axis++)if(abs(rd[axis])>1.e-9)distance=min(distance,max(travel[axis],0.));
 // Floor is deliberate: retain the last sample before a brick boundary.
 // The last partial segment is always visited. No new sample lattice is made.
 int next=int(floor((sampleT+distance-entry)/ds-fraction));
 spanEnd=max(index+1,min(next,count-1));
 return empty?spanEnd:index;
}
float segmentAlpha(float tau){return tau<1.e-3?tau*(1.-.5*tau+tau*tau/6.):1.-exp(-tau);}
float segmentIntegral(float tau,float len,float kappa){return tau<1.e-3?len*(1.-.5*tau+tau*tau/6.):(1.-exp(-tau))/kappa;}

#ifdef PARTICLE_EVENTS
layout(std430,binding=3) readonly buffer PC {uint particleCounts[];};
layout(std430,binding=5) readonly buffer PI {uint particleIndices[];};
layout(std430,binding=6) readonly buffer PP {vec4 particleProjected[];};
layout(std430,binding=7) buffer PO {uint particleOverflow[];};
uniform ivec2 tileDims;
uniform int tileCapacity;
#ifndef PARTICLE_EVENT_CAPACITY
#define PARTICLE_EVENT_CAPACITY 32
#endif
const int MAX_PARTICLE_EVENTS=PARTICLE_EVENT_CAPACITY;
void gatherParticles(vec2 pixel,vec3 rd,float lastT,out int count,out float depth[MAX_PARTICLE_EVENTS],out vec4 value[MAX_PARTICLE_EVENTS]){
 uint stableOrder[MAX_PARTICLE_EVENTS];
 count=0;int observed=0;ivec2 tile=clamp(ivec2(floor(pixel/16.)),ivec2(0),tileDims-1);int index=tile.x+tileDims.x*tile.y;
 for(int k=0;k<int(min(particleCounts[index],uint(tileCapacity)));k++){
  uint id=particleIndices[index*tileCapacity+k];vec4 p=particleProjected[2*id],col=particleProjected[2*id+1];
  float d2=dot(pixel-p.xy,pixel-p.xy)/(p.w*p.w);if(d2>16.)continue;
  float t=p.z/dot(rd,cameraForward);if(t>=lastT)continue;
  float alpha=col.a*exp(-.5*d2);if(alpha<1.e-8)continue;
  observed++;
  if(count>=MAX_PARTICLE_EVENTS){atomicMax(particleOverflow[0],uint(observed));continue;}
  // Stable tie-break: equal-depth fragments must not inherit atomic tile
  // insertion order. Workspace growth, tiles, and thread count cannot change
  // the alpha-compositing order of a frozen particle state.
  int j=count;while(j>0&&(depth[j-1]>t||(depth[j-1]==t&&stableOrder[j-1]>id))){depth[j]=depth[j-1];value[j]=value[j-1];stableOrder[j]=stableOrder[j-1];j--;}
  depth[j]=t;value[j]=vec4(col.rgb,alpha);stableOrder[j]=id;count++;
 }
}
void depositParticle(vec4 event,inout vec3 light,inout float Tr){light+=Tr*event.rgb*event.a;Tr*=1.-event.a;}
#endif

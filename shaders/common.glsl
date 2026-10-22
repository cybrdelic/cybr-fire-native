// Packed MAC field: (nx+1,ny+1,nz+1), component offsets are explicit.
// Cell fields: fuel kg/m3, oxygen deficit kg/m3, sensible energy J/m3, soot kg/m3.
layout(local_size_x=128) in;
uniform ivec3 dims;
uniform float spacing;
uniform float dt;
int index3(ivec3 p,ivec3 d){return (p.z*d.y+p.y)*d.x+p.x;}
ivec3 coord3(int n,ivec3 d){return ivec3(n%d.x,(n/d.x)%d.y,n/(d.x*d.y));}
int cells(ivec3 d){return d.x*d.y*d.z;}
vec3 faceOffset(int a){vec3 r=vec3(.5);r[a]=0.;return r;}
float smooth01(float x){x=clamp(x,0.,1.);return x*x*(3.-2.*x);}
uint hash32(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float rand01(uint x){return float(hash32(x)>>8)*(1./16777216.);}

// Flatten a 2-D dispatch without duplicating cell work beyond the X limit.
int invocationIndex(){return int(gl_GlobalInvocationID.x+gl_GlobalInvocationID.y*gl_NumWorkGroups.x*gl_WorkGroupSize.x);}

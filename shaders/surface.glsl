layout(binding=7) uniform sampler3D directTexture;
uniform vec3 keyDirection,keyColor;
uniform float keyIntensity,phaseG;
// 4*pi normalized gain; key intensity is a working angular-mean radiance scale.
float phaseGain(float mu){float den=max(1.+phaseG*phaseG-2.*phaseG*mu,.001);return (1.-phaseG*phaseG)/(den*sqrt(den));}
vec3 directScattering(vec3 uv,vec3 rd){return keyColor*keyIntensity*phaseGain(dot(normalize(keyDirection),rd))*textureLod(directTexture,uv,0.).r;}
vec3 surfaceRadiance(vec3 p,vec3 normal,vec3 view,int object){
 vec4 material=solidData[object*4+3];float roughness=solidData[object*4+2].w,metallic=material.w;
 vec3 uv=(p+normal*spacing)/(vec3(dims)*spacing),incident=textureLod(lightingTexture,uv,0.).rgb;
 vec3 light=normalize(keyDirection),halfVector=normalize(light+view);
 float nl=max(dot(normal,light),0.),nv=max(dot(normal,view),.001),nh=max(dot(normal,halfVector),0.),vh=max(dot(view,halfVector),0.);
 float a2=pow(roughness,4.),d=nh*nh*(a2-1.)+1.,D=a2/(3.14159265*d*d),k=pow(roughness+1.,2.)/8.;
 float G=(nv/(nv*(1.-k)+k))*(nl/(nl*(1.-k)+k));
 vec3 F0=mix(vec3(.04),material.rgb,metallic),F=F0+(1.-F0)*pow(1.-vh,5.);
 vec3 diffuse=material.rgb*(1.-metallic)/3.14159265,specular=D*G*F/(4.*nv*max(nl,.001));
 int shadowObject;float solidShadow=firstSolidHit(p+normal*.001,light,shadowObject)<1.e19?0.:1.;
 float visibility=solidShadow*textureLod(directTexture,uv,0.).r;
 return incident*(material.rgb*(1.-metallic)+F0)+keyColor*keyIntensity*3.14159265*nl*visibility*(diffuse+specular)+material.rgb*.008;
}

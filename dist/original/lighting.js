// Field-derived local lighting and analytic partial visibility.
export const lightFragment = `in vec2 uv;
out vec4 frag;
// Integrate power and its first spatial moment over one XZ quadrant.
// Quadrature cells widen away from the injector and near the plume outlet.
float radialEdge(float i){float t=i/7.;return .96*t*(.39+.61*t);}
float heightEdge(float i){float t=i/9.;return 2.32*t*(.31+.69*t);}
void main() {
    int sector=int(gl_FragCoord.x);
    vec2 quadrant=vec2((sector & 1)==0 ? -1. : 1.,(sector & 2)==0 ? -1. : 1.);
    vec3 moment=vec3(0.); float power=0.;
    for(int j=0;j<9;j++){
        float ya=heightEdge(float(j)), yb=heightEdge(float(j+1));
        for(int k=0;k<7;k++){
            float za=radialEdge(float(k)), zb=radialEdge(float(k+1));
            for(int i=0;i<7;i++){
                float xa=radialEdge(float(i)), xb=radialEdge(float(i+1));
                vec3 pos=.5*vec3(quadrant.x*(xa+xb),ya+yb,quadrant.y*(za+zb));
                vec4 state=sampleGrid(uField,pos);
                float temp=max(state.r,0.), t2=temp*temp;
                float radiance=t2*t2*(.48+1.75*min(state.g,.85))*smoothstep(.12,.38,temp);
                float dPower=radiance*((xb-xa)*(yb-ya)*(zb-za));
                power+=dPower; moment+=pos*dPower;
            }
        }
    }
    vec3 lightCenter=power>.0000001 ? moment/power : vec3(.14*quadrant.x,.26,.14*quadrant.y);
    frag=vec4(lightCenter,power*14.);
}
`;

export const lightHelper = `uniform highp sampler2D uLights;

// Cast a segment to an area emitter. The blocker is an inscribed capsule;
// penumbra grows with the fraction of the receiver-emitter distance.
float sfCapsuleVisibility(vec3 from,vec3 ray,float rr,vec3 a,vec3 b,float radius) {
    vec3 axis=b-a, offset=from-a;
    float vv=dot(axis,axis),rv=dot(ray,axis),ro=dot(ray,offset),vo=dot(axis,offset);
    float t=clamp((rv*vo-vv*ro)/max(.0000001,rr*vv-rv*rv),0.,1.);
    float s=clamp((vo+t*rv)/vv,0.,1.);
    t=clamp((s*rv-ro)/rr,0.,1.);
    vec3 delta=offset+t*ray-s*axis;
    float penumbra=.011+.16*t;
    return mix(.10,1.,smoothstep(max(0.,radius-penumbra),radius+penumbra,length(delta)));
}
float sfWoodShadow(vec3 p,vec3 ray,float rr,float logId) {
    if(dot(p.xz,p.xz)>2.9 || p.y>.35)return 1.;
    float visibility=1.;
    if(abs(logId-1.2)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(-.39,.070,-.11),vec3(.30,.066,-.17),.059);
    if(abs(logId-2.7)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(-.32,.068,.18),vec3(.39,.070,.085),.058);
    if(abs(logId-3.4)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(-.28,.076,-.31),vec3(.17,.087,.30),.056);
    if(abs(logId-4.1)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(-.20,.127,.30),vec3(.19,.176,-.32),.058);
    if(abs(logId-5.2)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(.38,.065,.24),vec3(-.22,.202,-.19),.039);
    if(abs(logId-6.9)>.05)visibility*=sfCapsuleVisibility(p,ray,rr,vec3(-.37,.045,.10),vec3(.24,.185,-.08),.033);
    return visibility;
}
struct SfIrradiance {vec3 direct;vec3 fill;vec3 specular;vec3 gloss;float flux;};
SfIrradiance sfIncident(vec3 p,vec3 n,vec3 view,bool wood,float logId){
    SfIrradiance light=SfIrradiance(vec3(0),vec3(0),vec3(0),vec3(0),0.);
    for(int i=0;i<4;i++){
        vec4 sector=texelFetch(uLights,ivec2(i,0),0);
        light.flux+=sector.w;
        vec3 ray=sector.xyz-(p+.003*n); float rr=max(.00001,dot(ray,ray));
        vec3 l=ray*inversesqrt(rr);
        float incidence=max(0.,dot(n,l));
        float visible=wood && incidence>0. ? sfWoodShadow(p+.003*n,ray,rr,logId) : 1.;
        vec3 e=vec3(1.,.341,.083)*sector.w/(.107+rr);
        if(uMode!=0)e*=.27;
        float reflectivity=max(0.,dot(reflect(-l,n),view));
        light.fill+=e;
        light.direct+=e*(incidence*visible);
        light.specular+=e*(pow(reflectivity,23.)*visible);
        light.gloss+=e*(pow(reflectivity,5.)*visible);
    }
    return light;
}
`;

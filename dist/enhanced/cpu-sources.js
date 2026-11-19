// Dimensionless source profiles shared by the reduced CPU volumes. Positions
// use the same world-space burner scale as the perspective view.
const profiles={
    fireball:{x:0,y:.49,radius:.36,height:.29,heat:1.09,fuel:.83,soot:.21,spread:2.10,lift:.97},
    groundburst:{x:-.30,y:.20,radius:.42,depth:.30,height:.135,heat:.98,fuel:.81,soot:.34,spread:2.25,lift:.28},
    embers:{x:0,y:.072,radius:.47,depth:.38,height:.10},
    pulsing:{x:0,y:.135,radius:.125,height:.11}
};
const unit=value=>Math.max(0,Math.min(1,value));
const smooth=value=>{const v=unit(value);return v*v*(3-2*v);};

export function cpuSource(preset) {
    return Object.hasOwn(profiles,preset)?profiles[preset]:null;
}

export function burstAmount(profile,time,dt){
    if(!profile?.spread)return 0;
    // Exact integral of a smooth, finite deposition. Several time steps
    // consume the pulse without creating a solver-rate-dependent fireball.
    const start=.10,duration=.16;
    return smooth((time+dt-start)/duration)-smooth((time-start)/duration);
}

export function sourceShape(preset,profile,x,y,z=0) {
    const rx=x-profile.x, ry=y-profile.y;
    const density=rx*rx/profile.radius**2+z*z/(profile.depth??profile.radius)**2+ry*ry/profile.height**2;
    if(profile.spread){
        const rough=.18*Math.sin(16*x-11*y+17*z)*Math.cos(13*x+14*y-15*z)+.11*Math.cos(28*x+11*y+9*z);
        const envelope=1-smooth((density+rough-.26)/1.42);
        const folds=.79+.21*Math.sin(17*x+14*y-20*z)*Math.cos(25*x-13*y+13*z);
        return Math.max(0,envelope*folds);
    }
    const variation=preset==='embers'?.55+.45*Math.sin(21*x+4*z)*Math.sin(13*x-18*z):1;
    return variation*Math.exp(-density);
}

export function continuousSource(preset,time) {
    if(preset==='embers')return {feed:.53,heat:.88,soot:2.0,lift:.26};
    if(preset==='pulsing'){
        const phase=(time+0.30)%1.72;
        const burst=smooth((phase-.31)/.15)*(1-smooth((phase-.64)/.24));
        return {feed:.018+4.1*burst,heat:.77,soot:.25,lift:.40+3.6*burst};
    }
    return null;
}

export function emberDraft(x,z,time){
    const wave=Math.max(0,Math.sin(3.55*time+8*x+11*z)*Math.sin(4.79*time-14*x+5*z));
    return .17+2.65*wave**3;
}

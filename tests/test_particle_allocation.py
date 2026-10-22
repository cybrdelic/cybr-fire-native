#!/usr/bin/env python3
"""Crowded particle workspaces retry without loss, mutation, or scene retuning."""
from pathlib import Path
from dataclasses import replace
import json,sys,time,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cybrfire.config import example,Embers
from cybrfire.gpu import ComputeContext
from cybrfire.solver import Simulation,SHADERS
from cybrfire.render import VolumeRenderer
from cybrfire.particle_render import grown_capacity
results=[]
def check(name,ok,**data):
    results.append(dict(test=name,passed=bool(ok),**data))
    print(('PASS ' if ok else 'FAIL ')+name+' '+json.dumps(data),flush=True)
def main():
    start=time.perf_counter();ctx=ComputeContext();base=example('burner','test')
    cfg=replace(base.render,width=160,height=160,samples=1,floor=False,ambient=0.,key_intensity=0.,
        background=(0.,0.,0.),camera_height=1.,target_height=1.,yaw=0.,camera_distance=4.,
        scatter_albedo=0.,bloom=0.,transmittance_cutoff=0.,empty_space_skipping=False,ember_tile_capacity=16)
    sim=Simulation(ctx,replace(base,domain=replace(base.domain,pressure_backend='spectral'),render=cfg,
        embers=Embers(enabled=True,rate_per_emitter=0.,capacity=512)))
    a=np.zeros(sim.embers.state.shape,np.float32)
    scale=160/(2*math.tan(math.radians(cfg.fov)/2)*4.)
    k=0
    for y in range(13):
        for x in range(13):
            a[k,0]=(.96+(x+.5)/scale,1.-(y+.5)/scale,.96,.002)
            a[k,1,3]=2200.+k*.5;k+=1
    sim.embers.state.write(a);before=sim.snapshot()
    auto=VolumeRenderer(ctx,sim);image=auto.render_hdr()
    check('crowded_tile_reproduces_original_128_limit',auto.particles.max_tile_occupancy>128,required=auto.particles.max_tile_occupancy)
    check('tile_allocation_grows_to_measured_demand',auto.particles.shape[2]>=auto.particles.max_tile_occupancy and auto.particles.tile_growths==1,capacity=auto.particles.shape[2],growths=auto.particles.tile_growths)
    ref=VolumeRenderer(ctx,sim,replace(cfg,ember_tile_capacity=512));reference=ref.render_hdr()
    error=float(np.max(np.abs(reference-image)))
    check('adaptive_tiles_match_generous_reference',error<2e-6,maximum_HDR_error=error)
    auto.render_hdr(128,128,prepare=False)
    check('allocation_retains_high_water_after_resize',auto.particles.shape[2]>=256,capacity=auto.particles.shape[2])
    # A deliberately overlapping stack exercises the separate sample-event limit.
    a.fill(0.)
    for j in range(40):
        a[j,0]=(.96,1.,.90+.002*j,.008);a[j,1,3]=2100.+j
    sim.embers.state.write(a)
    dense=VolumeRenderer(ctx,sim);overlap=dense.render_hdr()
    check('sample_event_capacity_grows_from_actual_overlap',dense.particle_event_capacity>=40 and dense.particle_event_growths==1,capacity=dense.particle_event_capacity,growths=dense.particle_event_growths)
    generous=VolumeRenderer(ctx,sim,replace(cfg,ember_tile_capacity=512))
    generous.march=ctx.program(SHADERS/'render.comp',{'PARTICLE_EVENTS':1,'PARTICLE_EVENT_CAPACITY':64})
    generous.particle_event_capacity=64;generous.particle_event_variants[64]=generous.march
    correct=generous.render_hdr();err=float(np.max(np.abs(correct-overlap)))
    check('retried_events_match_no_overflow_reference',err<2e-6,maximum_HDR_error=err)
    snap=sim.snapshot();dense.render_hdr(prepare=False)
    check('render_retries_do_not_mutate_numerical_state',np.array_equal(snap['particles'],sim.embers.state.read()) and np.array_equal(before['q'],sim.q.read()) and sim.tick==before['tick'])
    check('workspace_growth_bounded_by_particle_capacity',grown_capacity(129,160)==160 and grown_capacity(17,512)==32)
    for r in (auto,ref,dense,generous):r.close()
    ctx.close();report=dict(passed=sum(x['passed'] for x in results),failed=sum(not x['passed'] for x in results),seconds=time.perf_counter()-start,checks=results)
    (ROOT/'evidence/particle-allocation-tests.json').write_text(json.dumps(report,indent=2));return int(report['failed']>0)
if __name__=='__main__':sys.exit(main())

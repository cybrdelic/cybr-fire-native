"""Bounded-memory diagnostics and checkpoints for large image fields."""
from pathlib import Path
import json,zipfile
import numpy as np

def diagnostics(sim,slab_depth=8):
    nx,ny,nz=sim.shape
    low=np.full(4,np.inf);high=-low.copy();integrals=np.zeros(4);finite=True
    solid_max=0.
    for z in range(0,nz,slab_depth):
        d=min(slab_depth,nz-z);a=sim.q.read_slab(z,d)
        low=np.minimum(low,a.min(axis=(0,1,2)));high=np.maximum(high,a.max(axis=(0,1,2)))
        integrals+=a.sum(axis=(0,1,2),dtype=np.float64)
        finite=finite and bool(np.isfinite(a).all())
        if sim.scene.solids:
            selected=a[~sim.solids.fluid[z:z+d]]
            if selected.size:solid_max=max(solid_max,float(abs(selected).max()))
    max_speed=0.;floor_normal=0.;divsum=0.;divmax=0.
    for z in range(0,nz,slab_depth):
        d=min(slab_depth,nz-z);v=sim.vel.read_slab(z,d+1)
        finite=finite and bool(np.isfinite(v).all())
        max_speed=max(max_speed,float(np.abs(v[...,:3]).max()))
        floor_normal=max(floor_normal,float(np.abs(v[:,0,:,1]).max()))
        div=(v[:-1,:-1,1:,0]-v[:-1,:-1,:-1,0]+v[:-1,1:,:-1,1]-v[:-1,:-1,:-1,1]+v[1:,:-1,:-1,2]-v[:-1,:-1,:-1,2])/sim.h
        divmax=max(divmax,float(np.abs(div).max()));divsum+=np.square(div.astype(np.float64)).sum()
    result={'time':sim.time,'tick':sim.tick,'finite':finite,'min_species':low.tolist(),'max_species':high.tolist(),
        'integrals':(integrals*sim.h**3).tolist(),'max_temperature_K':float(300+high[2]/sim.scene.chemistry.heat_capacity),
        'max_component_speed':max_speed,'divergence_rms':float(np.sqrt(divsum/sim.n)),
        'divergence_max':divmax,'cfl_component':float(max_speed*sim.scene.domain.tick/sim.h),
        'floor_normal_max':floor_normal,'diagnostic_readback':'bounded z slabs'}
    if sim.embers:result['embers']=sim.embers.diagnostics()
    if sim.scene.solids:result.update(solid_scalar_max=solid_max,pressure_solver=sim.mg.last)
    if not finite or low.min()<-1e-4 or high[1]>sim.scene.chemistry.oxygen_ambient+1e-5:
        sim.fault='Nonfinite or inadmissible species detected';raise RuntimeError(sim.fault)
    return result

def save_checkpoint(sim,path,slab_depth=8):
    if sim.fault:raise RuntimeError('Cannot checkpoint a faulted simulation')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for name,field in [('q',sim.q),('velocity',sim.vel)]:
                with z.open(name+'.npy','w',force_zip64=True) as f:
                    np.lib.format.write_array_header_1_0(f,dict(descr=np.dtype(np.float32).str,fortran_order=False,shape=field.shape))
                    for k in range(0,field.grid_shape[2],slab_depth):
                        a=field.read_slab(k,min(slab_depth,field.grid_shape[2]-k));f.write(a.tobytes())
            metadata={'pressure':sim.mg.pressure.read(),'tick':np.asarray(sim.tick),'version':np.asarray(7),
                'config_hash':np.asarray(sim.config_hash),'config':np.asarray(json.dumps(sim.scene.to_dict(),sort_keys=True))}
            if sim.embers:metadata['particles']=sim.embers.state.read()
            for name,values in metadata.items():
                with z.open(name+'.npy','w') as f:np.save(f,values,allow_pickle=False)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True);raise

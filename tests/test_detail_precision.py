"""Additional CPU checks of cache precision and floor-window curl construction."""
from pathlib import Path
import argparse,json,mmap,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cybrfire.detail import DetailConfig,transverse_modes
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=root/'cache/parent');a=p.parse_args()
checks=[]
def check(name,passed,**data):
 checks.append(dict(name=name,passed=bool(passed),**data));print('PASS' if passed else 'FAIL',name,data,flush=True)
m=transverse_modes(DetailConfig()).astype(np.float64)
def value(p):
 k=m[::2,:3];pol=m[1::2,:3];ph=k@p+m[::2,3]-.12*m[1::2,3]
 u=np.clip(p[1]/.16,0,1);w=u*u*(3-2*u);dw=6*u*(1-u)/.16 if 0<p[1]<.16 else 0
 potential=np.cross(k,pol)/np.sum(k*k,axis=1)[:,None]*np.cos(ph)[:,None]
 return np.sum(w*pol*np.sin(ph)[:,None]+np.cross([0,dw,0],potential),axis=0)
points=[np.array([.2,y,.45]) for y in (.001,.017,.053,.09,.155,.2,.5)]
errors=[]
for point in points:
 h=2e-6;deriv=[]
 for axis in range(3):
  delta=np.eye(3)[axis]*h;deriv.append((value(point+delta)[axis]-value(point-delta)[axis])/(2*h))
 errors.append(abs(sum(deriv)))
check('floor_taper_retains_curl_divergence_cancellation',max(errors)<1e-4,maximum_finite_difference_divergence=float(max(errors)))
check('floor_modes_vanish_on_boundary',np.array_equal(value(np.array([.2,0.,.45])),np.zeros(3)))
manifest=json.loads((a.cache/'manifest.json').read_text());dims=tuple(n+1 for n in manifest['grid'])
v=np.load(a.cache/'velocity-000.npy',mmap_mode='r',allow_pickle=False)
square=0.;ref_square=0.;maximum=0.;finite=True;count=0
for z in range(0,dims[2],2):
 lo=z*dims[0]*dims[1];hi=min(z+2,dims[2])*dims[0]*dims[1]
 sample=np.array(v[lo:hi,:3],copy=True);rounded=sample.astype(np.float16).astype(np.float32);error=rounded-sample
 maximum=max(maximum,float(np.max(np.abs(error))));square+=float(np.sum(error*error,dtype=np.float64));ref_square+=float(np.sum(sample*sample,dtype=np.float64));count+=error.size
 finite=finite and bool(np.isfinite(rounded).all());v._mmap.madvise(mmap.MADV_DONTNEED)
relative=(square/ref_square)**.5
check('initial_parent_cache_FP16_conversion_finite',finite)
check('initial_parent_cache_FP16_relative_L2',relative<.001,relative_L2=float(relative),maximum_component_error_m_s=maximum,components_checked=count)
report=dict(passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),checks=checks,qualification='CPU conversion of the first parent snapshot, not full GPU-cache or trajectory parity. The curl test samples the analytic modeled field before texture reconstruction.')
(root/'evidence-v8/detail-precision-tests.json').write_text(json.dumps(report,indent=2));raise SystemExit(bool(report['failed']))

"""Shared analytic solids, voxel free-slip flow boundaries and hybrid PCG."""
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg,LinearOperator
from scipy.fft import dst,idst,dct,idct
from scipy.ndimage import label

def signed_distance(points,solid):
    p=np.asarray(points,dtype=np.float64)-np.asarray(solid.center)
    if solid.kind=='sphere':return np.linalg.norm(p,axis=-1)-solid.radius
    if solid.kind=='box':
        q=np.abs(p)-solid.half_extents
        return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
    ba=np.asarray(solid.end)-np.asarray(solid.center);t=np.clip(np.sum(p*ba,axis=-1)/np.dot(ba,ba),0,1)
    return np.linalg.norm(p-t[...,None]*ba,axis=-1)-solid.radius

class SolidSet:
    def __init__(self,ctx,shape,spacing,solids,native_fields=False):
        self.ctx,self.shape,self.h,self.solids=ctx,tuple(shape),spacing,tuple(solids);records=[]
        for s in solids:
            kind={'sphere':0,'box':1,'capsule':2}[s.kind]
            records.extend([(*s.center,float(kind)),(*s.half_extents,s.radius),(*s.end,s.roughness),(*s.albedo,s.metallic)])
        self.records=ctx.buffer((max(len(records),1),4),data=np.asarray(records,np.float32) if records else np.zeros((1,4),np.float32))
        self.fluid=None;self.mask=None;self.kernel=None
        if solids:
            nx,ny,nz=shape;z,y,x=np.indices((nz,ny,nx),dtype=np.float32)
            points=np.stack((x+.5,y+.5,z+.5),axis=-1)*spacing;fluid=np.ones((nz,ny,nx),bool)
            for s in solids:fluid &= signed_distance(points,s)>0
            if np.all(fluid):raise ValueError('Solid set unresolved at selected grid')
            if np.count_nonzero(fluid)<fluid.size*.3:raise ValueError('Less than 30% fluid remains')
            components,count=label(fluid)
            connected=np.unique(np.concatenate((components[0].ravel(),components[-1].ravel(),components[:,-1].ravel(),components[:,:,0].ravel(),components[:,:,-1].ravel())))
            if any(i not in connected for i in range(1,count+1)):raise ValueError('Sealed cavity requires an unsupported pressure gauge')
            self.fluid=fluid;self.mask=ctx.buffer(fluid.size,data=fluid.ravel().astype(np.float32))
            self.kernel=ctx.program(Path(__file__).resolve().parents[1]/'shaders/solid_boundary.comp',{'NATIVE_FIELDS':1} if native_fields else None)
    def bind(self,binding=4):self.ctx.glBindBufferBase(0x90D2,binding,self.records.handle)
    def enforce(self,q,velocity,operation=0):
        if self.mask is not None:self.kernel.run(np.asarray(self.shape)+1,[q,velocity,self.mask],dims=self.shape,operation=operation)
    def close(self):
        self.records.close()
        if self.mask is not None:self.mask.close()

def masked_matrix(fluid):
    """SPD graph Laplacian with Neumann solid/floor and odd-ghost open faces."""
    shape=fluid.shape;n=fluid.size;ids=np.arange(n,dtype=np.int32).reshape(shape)
    diag=np.zeros(shape,np.float32);rows=[];cols=[];data=[]
    for a in range(3):
        lo=[slice(None)]*3;hi=lo.copy();lo[a]=slice(None,-1);hi[a]=slice(1,None);lo,hi=tuple(lo),tuple(hi)
        valid=fluid[lo]&fluid[hi];i=ids[lo][valid];j=ids[hi][valid]
        rows.extend((i,j));cols.extend((j,i));data.extend((-np.ones(i.size,np.float32),-np.ones(j.size,np.float32)))
        diag[lo]+=valid;diag[hi]+=valid
        for end in (0,-1):
            if a==1 and end==0:continue
            f=[slice(None)]*3;f[a]=end;f=tuple(f);diag[f]+=2*fluid[f]
    diag[~fluid]=1
    if np.any(diag==0):raise ValueError('Unconstrained pressure row')
    rows.append(ids.ravel());cols.append(ids.ravel());data.append(diag.ravel())
    return sparse.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(n,n)).tocsr()

class MaskedPressure:
    def __init__(self,ctx,shape,solids,rtol=2.e-5,maxiter=120):
        self.ctx,self.shape,self.solids=ctx,tuple(shape),solids;self._p=ctx.buffer(np.prod(shape));self._rhs=ctx.buffer(np.prod(shape))
        self.A=masked_matrix(solids.fluid);self.rtol,self.maxiter=rtol,maxiter;self.fluid=solids.fluid.ravel();nx,ny,nz=shape
        lx=4*np.sin(np.pi*(np.arange(nx)+1)/(2*nx))**2;ly=4*np.sin(np.pi*(np.arange(ny)+.5)/(2*ny))**2;lz=4*np.sin(np.pi*(np.arange(nz)+1)/(2*nz))**2
        self.eigen=(lz[:,None,None]+ly[None,:,None]+lx[None,None,:]).astype(np.float32)
        self.M=LinearOperator(self.A.shape,matvec=self.precondition,dtype=np.float32);self.last={'iterations':0,'relative_residual':0.}
    @property
    def pressure(self):return self._p
    @property
    def rhs(self):return self._rhs
    def precondition(self,b):
        original=np.asarray(b).ravel();a=np.where(self.fluid,original,0).reshape(self.shape[::-1])
        a=dst(a,type=2,axis=2,norm='ortho',workers=4);a=dst(a,type=2,axis=0,norm='ortho',workers=4)
        a=dct(a,type=4,axis=1,norm='ortho',workers=4)/self.eigen
        a=idct(a,type=4,axis=1,norm='ortho',workers=4);a=idst(a,type=2,axis=0,norm='ortho',workers=4);a=idst(a,type=2,axis=2,norm='ortho',workers=4)
        return np.where(self.fluid,a.ravel(),original)
    def solve(self,cycles=1,reset=False):
        b=self._rhs.read();b[~self.fluid]=0;count=0
        def cb(x):
            nonlocal count
            count+=1
        x,info=cg(self.A,b,x0=None if reset else self._p.read(),M=self.M,rtol=self.rtol,atol=1.e-8,maxiter=self.maxiter,callback=cb)
        res=float(np.linalg.norm((self.A@x-b).astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1.e-12))
        self.last={'iterations':count,'relative_residual':res,'converged':bool(info==0)}
        if info or (np.linalg.norm(b)>1.e-6 and res>self.rtol*2):raise RuntimeError('Masked pressure failed acceptance: '+str(self.last))
        x[~self.fluid]=0;self._p.write(x)
    def residual(self):return self._rhs.read()-self.A@self._p.read()

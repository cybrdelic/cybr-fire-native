"""Optional exact box Poisson reference for CPU/software-adapter runs.

This is an explicit HYBRID path: RHS readback -> CPU FFT -> pressure upload.
The all-compute multigrid backend remains independently executable and tested.
"""
import numpy as np
from scipy.fft import dst,idst,dct,idct

class SpectralPressure:
    def __init__(self,ctx,shape,workers=4):
        self.ctx=ctx;self.shape=tuple(shape);self.workers=workers;n=int(np.prod(shape))
        self._p=ctx.buffer(n);self._rhs=ctx.buffer(n)
        nx,ny,nz=shape
        lx=4*np.sin(np.pi*(np.arange(nx)+1)/(2*nx))**2
        ly=4*np.sin(np.pi*(np.arange(ny)+.5)/(2*ny))**2
        lz=4*np.sin(np.pi*(np.arange(nz)+1)/(2*nz))**2
        self.eigen=(lz[:,None,None]+ly[None,:,None]+lx[None,None,:]).astype(np.float32)
    @property
    def pressure(self):return self._p
    @property
    def rhs(self):return self._rhs
    def solve(self,cycles=1,reset=False):
        nx,ny,nz=self.shape
        a=self._rhs.read().reshape(nz,ny,nx)
        a=dst(a,type=2,axis=2,norm='ortho',workers=self.workers)
        a=dst(a,type=2,axis=0,norm='ortho',workers=self.workers)
        a=dct(a,type=4,axis=1,norm='ortho',workers=self.workers)
        a/=self.eigen
        a=idct(a,type=4,axis=1,norm='ortho',workers=self.workers)
        a=idst(a,type=2,axis=0,norm='ortho',workers=self.workers)
        a=idst(a,type=2,axis=2,norm='ortho',workers=self.workers)
        self._p.write(a.ravel())
    def residual(self):
        nx,ny,nz=self.shape;p=self._p.read().reshape(nz,ny,nx);A=np.zeros_like(p)
        for axis in range(3):
            for sign in (-1,1):
                neighbor=np.roll(p,-sign,axis=axis);face=[slice(None)]*3;face[axis]=0 if sign==-1 else -1
                neighbor[tuple(face)]=p[tuple(face)] if axis==1 and sign==-1 else -p[tuple(face)]
                A+=p-neighbor
        return self._rhs.read()-A.ravel()

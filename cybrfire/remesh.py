"""Conservative cell-average initial-condition remeshing, not render upsampling.
Limited linear reconstruction preserves integrals without creating extrema.
Velocity is interpolated at physical MAC faces and must then be reprojected.
"""
import numpy as np
from scipy.sparse import csr_matrix

def _weights(nold,nnew):
    rows=[];cols=[];weights=[];moments=[];width=nold/nnew
    for j in range(nnew):
        a=j*width;b=(j+1)*width
        for i in range(max(0,int(np.floor(a))),min(nold,int(np.ceil(b)))):
            lo=max(a,i);hi=min(b,i+1)
            if hi<=lo:continue
            rows.append(j);cols.append(i);weights.append((hi-lo)/width)
            moments.append(.5*((hi-i-.5)**2-(lo-i-.5)**2)/width)
    return csr_matrix((weights,(rows,cols)),shape=(nnew,nold)),csr_matrix((moments,(rows,cols)),shape=(nnew,nold))

def conservative_axis(array,size,axis):
    a=np.moveaxis(array,axis,0);old=a.shape[0];shape=(size,)+a.shape[1:]
    if old==size:return array.copy()
    flat=np.ascontiguousarray(a.reshape(old,-1));slopes=np.zeros_like(flat)
    dl=flat[1:-1]-flat[:-2];dr=flat[2:]-flat[1:-1]
    slopes[1:-1]=np.where(dl*dr>0,np.sign(dl)*np.minimum(np.abs(dl),np.abs(dr)),0.)
    W,M=_weights(old,size);out=(W@flat+M@slopes).astype(array.dtype)
    return np.moveaxis(out.reshape(shape),0,axis)

def conservative_scalars(array,new_shape):
    result=np.asarray(array)
    for axis,size in enumerate(new_shape[::-1]):result=conservative_axis(result,size,axis)
    return np.ascontiguousarray(result)

def _linear_axis(array,coordinates,axis):
    c=np.clip(coordinates,0,array.shape[axis]-1);lo=np.floor(c).astype(int);hi=np.minimum(lo+1,array.shape[axis]-1)
    t=c-lo;shape=[1]*array.ndim;shape[axis]=len(c);t=t.reshape(shape)
    return (np.take(array,lo,axis=axis)*(1-t)+np.take(array,hi,axis=axis)*t).astype(array.dtype)

def staggered_velocity(array,old_shape,new_shape):
    out=np.zeros(tuple(n+1 for n in new_shape[::-1])+(4,),dtype=np.float32)
    for component in range(3):
        old_dims=list(old_shape);new_dims=list(new_shape);old_dims[component]+=1;new_dims[component]+=1
        a=array[:old_dims[2],:old_dims[1],:old_dims[0],component]
        for axis in range(3):
            world_axis=2-axis;offset=0. if world_axis==component else .5
            coordinates=(np.arange(new_dims[world_axis])+offset)*old_shape[world_axis]/new_shape[world_axis]-offset
            a=_linear_axis(a,coordinates,axis)
        out[:new_dims[2],:new_dims[1],:new_dims[0],component]=a
    out[:,0,:,1]=0.;return out

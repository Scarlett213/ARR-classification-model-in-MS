import numpy as np
import nibabel as nib


def transs(prd,axis):
    a=[]
    if axis==2:
        a=np.zeros([121,145,prd.shape[2]])
        for i in range(prd.shape[axis]):
            hh=np.zeros([145,128])
            hh[:144,:]=prd[:,:,i]
            t=hh[:,:121]
            a[:,:,i]=t.T
    if axis==1:
        a=np.zeros([121,prd.shape[1],121])
        for i in range(prd.shape[axis]):
            hh=np.zeros([128,128])
            hh=prd[:,i,:]
            t=hh[:121,:121]
            a[:,i,:]=t.T
    if axis==0:
        a=np.zeros([prd.shape[0],145,121])
        for i in range(prd.shape[axis]):
            hh=np.zeros([128,145])
            hh[:,:144]=prd[i,:,:]
            t=hh[:121,:]
            a[i,:,:]=t.T
    return a

def MV(x, y, z):
    hh = (x + y + z) / 3
    hh[hh >= 0.5] = 1

    hh[hh < 0.5] = 0
    return hh

def add0(p,axis,num):
    be=min(num)
    en=max(num)+1
    aa=np.zeros([121,145,121])
    if axis==2:
        for i in range(len(num)):
            aa[:,:,num[i]]=p[:,:,i]
    if axis==1:
        for i in range(len(num)):
            aa[:,num[i],:]=p[:,i,:]
    if axis==0:
        for i in range(len(num)):
            aa[num[i],:,:]=p[i,:,:]
    return aa


def save_nii(data, path, affine):
    data=data.astype(np.float64)
    nib.save(nib.Nifti1Image(data, affine), path)


def loadNii(path):
    x=nib.load(path)
    a=x.get_affine()
    x=x.get_fdata()
    x=np.array(x)
    return x,a
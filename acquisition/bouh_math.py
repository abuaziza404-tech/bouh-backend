"""Numerical backend, compatible with compute_indices(arrays, mask, numpy).

Ratios are spectral proxies; they do not identify gold, grade or burial depth.
"""
import math


def calibration(asset):
    bands=asset.get('raster:bands',[])
    if len(bands)!=1 or 'scale' not in bands[0] or 'offset' not in bands[0]:
        raise ValueError('MISSING_EXPLICIT_SCALE_OR_OFFSET')
    scale,offset=bands[0]['scale'],bands[0]['offset']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (scale,offset)) or scale<=0:
        raise ValueError('INVALID_RADIOMETRIC_CALIBRATION')
    return float(scale),float(offset)


def safe_divide(num,den,np,mask=None,epsilon=1e-6):
    a,b=np.broadcast_arrays(np.asarray(num,dtype='float64'),np.asarray(den,dtype='float64'))
    valid=np.isfinite(a)&np.isfinite(b)&(np.abs(b)>epsilon)
    if mask is not None:valid &= mask
    out=np.full(a.shape,np.nan,dtype='float64')
    with np.errstate(all='ignore'):np.divide(a,b,out=out,where=valid)
    valid &= np.isfinite(out)&(np.abs(out)<=np.finfo('float32').max)
    return np.where(valid,out,np.nan).astype('float32')


def compute_indices(arrays,mask,numpy,ndvi_max=.25):
    np=numpy;required=['blue','green','red','nir','swir16','swir22']
    if mask.dtype!=np.bool_ or any(k not in arrays or arrays[k].shape!=mask.shape for k in required):
        raise ValueError('MISMATCHED_GRIDS_OR_MASK_TYPE')
    valid=mask.copy()
    for k in required:valid &= np.isfinite(arrays[k])&(arrays[k]>=0)
    ndvi=safe_divide(arrays['nir']-arrays['red'],arrays['nir']+arrays['red'],np,valid)
    ndwi=safe_divide(arrays['green']-arrays['nir'],arrays['green']+arrays['nir'],np,valid)
    bare=valid&np.isfinite(ndvi)&np.isfinite(ndwi)&(ndvi<=ndvi_max)&(ndwi<=0)
    outputs={'ndvi':ndvi,'ndwi':ndwi}
    for name,num,den in [('iron_ratio','red','blue'),('swir_ratio','swir16','swir22'),('gossan_ratio','swir16','red')]:
        outputs[name]=safe_divide(arrays[num],arrays[den],np,bare)
    # Valid-mask is per ratio: a missing denominator never becomes index zero.
    outputs['valid_bare_mask']=bare.astype('uint8')
    return outputs

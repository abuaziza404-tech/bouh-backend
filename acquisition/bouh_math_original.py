"""Numerical checks independent of network/raster IO. No gold probability model."""
import math

def calibration(asset):
    bands = asset.get('raster:bands', [])
    if len(bands) != 1 or 'scale' not in bands[0] or 'offset' not in bands[0]:
        raise ValueError('MISSING_EXPLICIT_SCALE_OR_OFFSET')
    scale, offset = bands[0]['scale'], bands[0]['offset']
    if isinstance(scale,bool) or isinstance(offset,bool) or not isinstance(scale,(int,float)) or not isinstance(offset,(int,float)) or not math.isfinite(scale) or not math.isfinite(offset) or scale <= 0:
        raise ValueError('INVALID_RADIOMETRIC_CALIBRATION')
    return float(scale), float(offset)

def compute_indices(arrays, mask, numpy):
    np=numpy
    required=['blue','red','nir','swir16','swir22']
    shape=mask.shape
    if any(k not in arrays or arrays[k].shape != shape for k in required):
        raise ValueError('MISMATCHED_GRIDS')
    valid=mask.copy()
    for k in required: valid &= np.isfinite(arrays[k]) & (arrays[k]>0)
    valid &= arrays['nir']+arrays['red']>1e-6
    ndvi=np.full(shape,np.nan,dtype='float32')
    np.divide(arrays['nir']-arrays['red'],arrays['nir']+arrays['red'],out=ndvi,where=valid)
    bare=valid & (ndvi<0.25)
    outputs={}
    for name,numer,denom in [('swir_ratio',arrays['swir16'],arrays['swir22']),('iron_ratio',arrays['red'],arrays['blue'])]:
        ok=bare & (denom>1e-6)
        out=np.full(shape,np.nan,dtype='float32')
        np.divide(numer,denom,out=out,where=ok)
        outputs[name]=out
    outputs['ndvi']=ndvi
    outputs['valid_bare_mask']=bare.astype('uint8')
    return outputs

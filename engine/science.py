"""Pure NumPy geology/PCA/terrain routines; no network side effects."""
import math
import numpy as np
from .numeric import select_component,QualityGateError


def safe_divide(a,b,mask=None,epsilon=1e-6):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    ok=np.isfinite(a)&np.isfinite(b)&(np.abs(b)>epsilon)
    if mask is not None:ok &= mask
    out=np.full(a.shape,np.nan,dtype=float)
    with np.errstate(all='ignore'):np.divide(a,b,out=out,where=ok)
    out[~np.isfinite(out)]=np.nan
    return out


def terrain_light(dem,resolution,azimuth,elevation,max_distance=6000):
    """Slope/aspect self-shadow AND finite-distance cast shadows.

    DEM in a local metric north-up grid, not angular degrees. Ray samples march
    toward the sun. Missing or out-of-halo terrain makes illumination unknown;
    a DEM halo >=max_distance+2*resolution is mandatory around the AOI.
    Resolution30m does not resolve smaller obstacles; no radiometric correction.
    """
    if not(0<=azimuth<=360 and 5<=elevation<90 and resolution>0 and max_distance>0):
        raise ValueError('INVALID_SOLAR_OR_TERRAIN_PARAMETERS')
    d=np.asarray(dem,dtype=float)
    gy,gx=np.gradient(d,resolution,resolution)
    slope=np.arctan(np.hypot(gx,gy))
    aspect=np.arctan2(-gx,gy) # downhill direction clockwise from north
    az,el=np.deg2rad([azimuth,elevation])
    cos_i=np.cos(slope)*np.sin(el)+np.sin(slope)*np.cos(el)*np.cos(az-aspect)
    known=np.isfinite(d)&np.isfinite(cos_i)
    lit=known&(cos_i>.10)
    h,w=d.shape;seen=set()
    for k in range(1,int(math.ceil(max_distance/resolution))+1):
        dr=int(round(-math.cos(az)*k));dc=int(round(math.sin(az)*k))
        if (dr,dc) in seen:continue
        seen.add((dr,dc));dist=math.hypot(dr,dc)*resolution
        if dist==0:continue
        neighbor=np.full(d.shape,np.nan)
        r0=max(0,-dr);r1=min(h,h-dr);c0=max(0,-dc);c1=min(w,w-dc)
        if r1>r0 and c1>c0:neighbor[r0:r1,c0:c1]=d[r0+dr:r1+dr,c0+dc:c1+dc]
        finite=np.isfinite(neighbor);known &= finite
        lit &= finite & ((neighbor-d)<=dist*math.tan(el))
    return lit&known,known,cos_i


def feature_pca(arrays,mask,kind):
    spec={'iron':(['blue','red','nir','swir16'],'red','blue'),
          'hydroxyl':(['blue','nir','swir16','swir22'],'swir16','swir22')}
    bands,pos,neg=spec[kind]
    valid=mask.copy()
    for b in bands:valid &= np.isfinite(arrays[b])
    x=np.column_stack([arrays[b][valid] for b in bands])
    if len(x)<500:raise QualityGateError('INSUFFICIENT_PCA_PIXELS')
    mean=x.mean(axis=0);std=x.std(axis=0,ddof=1)
    if np.any(std<=1e-6):raise QualityGateError('CONSTANT_PCA_BAND')
    z=safe_divide(x-mean,std)
    cov=np.cov(z,rowvar=False)
    vals,vecs=np.linalg.eigh(cov);order=np.argsort(vals)[::-1]
    vals=vals[order];rows=vecs[:,order].T
    selected=select_component(vals,rows,bands,pos,neg)
    score=np.full(mask.shape,np.nan)
    score[valid]=z@np.asarray(selected.loadings)
    return score,{'bands':bands,'mean':mean.tolist(),'std':std.tolist(),
        'covariance':cov.tolist(),'eigenvalues':vals.tolist(),'choice':selected.record(),
        'method':'FEATURE_ORIENTED_CORRELATION_PCA_UNVALIDATED'}


def classify(arrays,mask):
    """0/1/2 = relative evidence priority, 255 = invalid/unknown.

    All criteria share an optical source; no independence/probability claim.
    Require ratio+feature-PC agreement; 'strong' further requires joint families.
    Cannot identify sandstone vs gossan uniquely or predict detector response.
    """
    n=arrays['nir'];r=arrays['red'];g=arrays['green']
    ndvi=safe_divide(n-r,n+r,mask);ndwi=safe_divide(g-n,g+n,mask)
    valid=mask&np.isfinite(ndvi)&np.isfinite(ndwi)&(ndvi<=.25)&(ndwi<=0)
    ratios={
      'iron_ratio':safe_divide(r,arrays['blue'],valid),
      'swir_ratio':safe_divide(arrays['swir16'],arrays['swir22'],valid),
      'gossan_ratio':safe_divide(arrays['swir16'],r,valid)}
    for image in ratios.values():valid &= np.isfinite(image)
    iron,im=feature_pca(arrays,valid,'iron')
    hydroxyl,hm=feature_pca(arrays,valid,'hydroxyl')
    thresholds={}
    def upper(image,name,q=90):
        values=image[valid&np.isfinite(image)]
        if len(values)<500 or np.std(values)<1e-8:raise QualityGateError('NO_STABLE_LOCAL_VARIANCE_'+name)
        t=float(np.percentile(values,q));thresholds[name]={'percentile':q,'threshold':t}
        return image>t
    f=upper(ratios['iron_ratio'],'iron_ratio')&upper(iron,'iron_pc')
    h=upper(ratios['swir_ratio'],'swir_ratio')&upper(hydroxyl,'hydroxyl_pc')
    cap=upper(ratios['gossan_ratio'],'gossan_ratio',95)
    output=np.full(mask.shape,255,dtype='uint8');output[valid]=0
    output[valid&(f|h)]=1;output[valid&f&h&cap]=2
    return output,{'pca':{'iron':im,'hydroxyl':hm},'thresholds':thresholds,
                  'class_legend':{'0':'LOW_RELATIVE_EVIDENCE','1':'MODERATE_RELATIVE_EVIDENCE',
                                  '2':'STRONGER_COMBINED_EVIDENCE','255':'NO_VALID_DATA'},
                  'gold_probability':None,'field_validated':False,
                  'limitations':['Shared optical inputs, not independent evidence',
                  'No lithological validation or red-bed exclusion','No placer hydrological modelling',
                  'Not a calibrated probability or detector-response model']}


def xyz_bounds(z,x,y):
    if not all(isinstance(v,int) for v in [z,x,y]) or not 0<=z<=20 or not(0<=x<2**z and 0<=y<2**z):
        raise ValueError('INVALID_XYZ_TILE')
    origin=20037508.342789244;span=2*origin/(2**z)
    left=-origin+x*span;top=origin-y*span
    return left,top-span,left+span,top


def colorize(classes):
    palette=np.zeros((256,4),dtype='uint8')
    palette[0]=[70,110,150,100];palette[1]=[30,180,150,175];palette[2]=[240,125,35,220]
    return palette[np.asarray(classes,dtype='uint8')]

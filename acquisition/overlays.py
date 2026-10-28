"""Georeferenced visual derivatives, not analysis rasters or target scores."""
from pathlib import Path
import numpy as np
from PIL import Image


def create_overlay(source,output,color):
    import rasterio
    from rasterio.warp import calculate_default_transform,reproject,transform_bounds
    from rasterio.enums import Resampling
    with rasterio.open(source) as src:
        # Leaflet's imageOverlay stretches in map pixel space. Warp to3857
        # BEFORE PNG creation; assigning only WGS84 corners to a UTM PNG is wrong.
        transform,w,h=calculate_default_transform(src.crs,'EPSG:3857',src.width,src.height,*src.bounds)
        # Bound both dimensions for mobile memory/transfer.
        if max(w,h)>1024:
            scale=max(w,h)/1024
            transform,w,h=calculate_default_transform(src.crs,'EPSG:3857',src.width,src.height,*src.bounds,
                                                       dst_width=max(1,int(w/scale)),dst_height=max(1,int(h/scale)))
        dest=np.full((h,w),np.nan,dtype='float32')
        reproject(source=rasterio.band(src,1),destination=dest,src_transform=src.transform,src_crs=src.crs,
                  src_nodata=src.nodata,dst_transform=transform,dst_crs='EPSG:3857',dst_nodata=float('nan'),
                  resampling=Resampling.nearest)
    valid=np.isfinite(dest)
    if valid.sum()<10:raise ValueError('NO_DISPLAY_PIXELS')
    lo,hi=np.percentile(dest[valid],[2,98])
    if not np.isfinite(lo+hi) or hi<=lo:raise ValueError('CONSTANT_DISPLAY_LAYER')
    rgba=np.zeros((h,w,4),dtype='uint8')
    strength=np.zeros((h,w),dtype='float32');strength[valid]=np.clip((dest[valid]-lo)/(hi-lo),0,1)
    for i,c in enumerate(color):rgba[:,:,i]=np.where(valid,c,0)
    rgba[:,:,3]=np.where(valid,35+strength*180,0).astype('uint8')
    Image.fromarray(rgba).save(output)
    left,bottom,right,top=rasterio.transform.array_bounds(h,w,transform)
    west,south,east,north=transform_bounds('EPSG:3857','EPSG:4326',left,bottom,right,top,densify_pts=21)
    return {'bounds':[[south,west],[north,east]],'projection':'EPSG:3857',
            'width':w,'height':h,'display_percentiles':[float(lo),float(hi)],
            'display_only':True,'file':Path(output).name}

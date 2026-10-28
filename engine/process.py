"""Isolated job: bounded grids, source QA, DSM shadows, PCA, ordinal evidence."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .coverage import load_areas,require_processable
from .io import search,read_cog,calibration,solar_geometry,S2,DEM
from .science import terrain_light,classify


def execute(area_id,out_dir,coverage_file):
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.features import geometry_mask
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    from pyproj import Transformer
    from shapely.geometry import shape,mapping
    from shapely.ops import transform as transform_shape
    import requests
    area=require_processable(load_areas(coverage_file)[area_id]);poly=shape(area['geometry'])
    if not poly.is_valid or poly.is_empty:raise ValueError('INVALID_AOI_TOPOLOGY')
    lon,lat=poly.centroid.x,poly.centroid.y
    crs=f"EPSG:{(32600 if lat>=0 else 32700)+min(60,int((lon+180)//6)+1)}"
    forward=Transformer.from_crs(4326,crs,always_xy=True)
    reverse=Transformer.from_crs(crs,4326,always_xy=True)
    projected=transform_shape(forward.transform,poly)
    if projected.area>150_000_000:raise ValueError('AOI_TOO_LARGE')
    left,bottom,right,top=projected.bounds
    res=20;left=math.floor(left/res)*res;top=math.ceil(top/res)*res
    w,h=math.ceil((right-left)/res),math.ceil((top-bottom)/res)
    if w*h>300000:raise ValueError('AOI_PIXEL_BUDGET_EXCEEDED_SPLIT_REGION')
    affine=from_origin(left,top,res,res)
    inside=geometry_mask([mapping(projected)],out_shape=(h,w),transform=affine,invert=True)
    # Keep native nominal DSM30m grid plus a full6000m ray-tracing halo.
    halo=6090;ds=30;dl=math.floor((left-halo)/ds)*ds;dt=math.ceil((top+halo)/ds)*ds
    dw=math.ceil((right+halo-dl)/ds);dh=math.ceil((dt-bottom+halo)/ds)
    if dw*dh>1500000:raise ValueError('DEM_PIXEL_BUDGET_EXCEEDED')
    da=from_origin(dl,dt,ds,ds)
    from shapely.geometry import box
    dem_geometry=mapping(transform_shape(reverse.transform,box(dl,dt-dh*ds,dl+dw*ds,dt)))
    session=requests.Session();session.headers['User-Agent']='BOUH-Tile-Engine/0.4'
    dem_items=search(session,DEM,dem_geometry,maximum=16)
    dem=np.full((dh,dw),np.nan,dtype='float32');dem_sources=[]
    for item in dem_items:
        asset=item.get('assets',{}).get('data') or item.get('assets',{}).get('dem')
        if not asset:raise ValueError('MISSING_DEM_ASSET')
        window,meta=read_cog(asset,crs,da,dw,dh,'dem')
        use=np.isfinite(window)&~np.isfinite(dem);dem[use]=window[use];dem_sources.append(meta)
    if not dem_sources or not np.isfinite(dem).any():raise ValueError('DEM_UNAVAILABLE')
    scenes=search(session,S2,area['geometry'])
    failures=[];attempted=0
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    for item in scenes:
        attempted+=1
        if attempted>3:break
        try:
            az,el=solar_geometry(item)
            lit,known,_=terrain_light(dem,ds,az,el,max_distance=6000)
            dmask=np.where(known,lit.astype('uint8'),255)
            shadow=np.full((h,w),255,dtype='uint8')
            reproject(dmask,shadow,src_transform=da,src_crs=crs,src_nodata=255,
                      dst_transform=affine,dst_crs=crs,dst_nodata=255,resampling=Resampling.nearest)
            assets=item.get('assets',{})
            if 'scl' not in assets:raise ValueError('MISSING_SCL')
            scl,_=read_cog(assets['scl'],crs,affine,w,h,'qa')
            valid=inside&(scl==5)&(shadow==1)
            arrays={};sources=[]
            for band in ['blue','green','red','nir','swir16','swir22']:
                asset=assets.get(band)
                if not asset:raise ValueError('MISSING_BAND_'+band)
                scale,offset=calibration(asset)
                data,meta=read_cog(asset,crs,affine,w,h)
                arrays[band]=data*scale+offset
                valid &= np.isfinite(arrays[band])&(arrays[band]>=0)
                sources.append(meta|{'scale':scale,'offset':offset,'band':band})
            fraction=float(valid.sum()/max(1,inside.sum()))
            if fraction<.35:raise ValueError('INSUFFICIENT_CLEAR_ILLUMINATED_AOI')
            classes,science=classify(arrays,valid)
            if np.sum(classes!=255)<500:raise ValueError('INSUFFICIENT_FINAL_PIXELS')
            profile=dict(driver='GTiff',width=w,height=h,count=1,dtype='uint8',crs=crs,
                         transform=affine,nodata=255,compress='deflate',tiled=True)
            path=out/'classes.tif'
            with rasterio.open(path,'w',**profile) as dst:dst.write(classes,1)
            with rasterio.open(path) as check:
                if check.crs.to_string()!=crs or check.nodata!=255:raise ValueError('OUTPUT_QA_FAILED')
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            manifest={'schema':'BOUH_TILES_0.4','area_id':area_id,'product_id':item['id'],
                'acquisition_datetime':item['properties'].get('datetime'),
                'processed_at':datetime.now(timezone.utc).isoformat(),'valid_fraction_before_pca':fraction,
                'final_valid_fraction':float(np.sum(classes!=255)/max(1,inside.sum())),
                'crs':crs,'resolution_m':20,'dem':{'type':'DSM','native_m':30,'sources':dem_sources,
                'shadow_method':'SLOPE_ASPECT_PLUS_DISCRETE_CAST_RAYS','ray_limit_m':6000,
                'solar_azimuth':az,'solar_elevation':el},'sources':sources,
                'science':science,'classes_sha256':digest,'failures':failures,
                'raw_sha256':None,'raw_sha256_reason':'Source COGs read by ranges, not fully downloaded'}
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
            return
        except Exception as exc:failures.append({'product_id':item.get('id'),'reason':str(exc)})
    (out/'failure.json').write_text(json.dumps(failures,indent=2))
    raise ValueError('NO_SCENE_PASSED_QA_PCA_AND_DEM_GATES')

"""Coverage-driven replacement for the old hardcoded-focus pilot.

A server worker reads original COG windows and calibrates reflectance BEFORE
calling bouh_math. Public Earth Search mode needs no satellite API secret.
"""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json
import math
import os
from .bouh_math import calibration,compute_indices
from .coverage import load_areas,require_processable

CATALOG='https://earth-search.aws.element84.com/v1'
COLLECTION='sentinel-2-c1-l2a'
BANDS=['blue','green','red','nir','swir16','swir22']


def stamp():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
def trusted_asset(url):
    u=urlparse(url or '')
    host=u.hostname or ''
    if (u.scheme!='https' or u.username or u.password or u.port not in (None,443) or
        not host.endswith('.s3.us-west-2.amazonaws.com') or not u.path.lower().endswith('.tif')):
        raise ValueError('UNEXPECTED_RASTER_ASSET_HOST_OR_FORMAT')
    return url


def discover(session,geometry,days=30,max_scenes=3):
    end=datetime.now(timezone.utc);start=end-timedelta(days=days)
    payload={'collections':[COLLECTION],'intersects':geometry,'limit':100,
             'datetime':start.isoformat().replace('+00:00','Z')+'/'+end.isoformat().replace('+00:00','Z')}
    response=session.post(CATALOG+'/search',json=payload,timeout=(10,45));response.raise_for_status()
    page=response.json()
    if page.get('type')!='FeatureCollection' or not isinstance(page.get('features'),list):
        raise ValueError('INVALID_STAC_RESPONSE')
    if any(l.get('rel')=='next' for l in page.get('links',[])):
        # Do not claim to have found the latest scene when the catalogue result
        # was incomplete and server-side datetime sorting is not established.
        raise ValueError('CATALOG_PAGINATION_REQUIRED_NARROW_LOOKBACK')
    items=sorted(page['features'],key=lambda x:str(x.get('properties',{}).get('datetime','')),reverse=True)
    unique=[];seen_dates=set()
    for item in items:
        date=str(item.get('properties',{}).get('datetime',''))[:10]
        if not date or date in seen_dates:continue
        seen_dates.add(date);unique.append(item)
        if len(unique)>=max_scenes:break
    return unique,payload


def collect(area_id,output,days=30,max_scenes=3,coverage_path='coverage-areas.geojson'):
    import requests
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_origin
    from rasterio.features import geometry_mask
    from pyproj import Transformer
    from shapely.geometry import shape,mapping
    from shapely.ops import transform as geom_transform
    area=require_processable(load_areas(coverage_path)[area_id])
    polygon=shape(area['geometry'])
    if not polygon.is_valid or polygon.is_empty:raise ValueError('INVALID_AOI_TOPOLOGY')
    w,s,e,n=area['bbox'];lon=(w+e)/2;lat=(s+n)/2
    epsg=(32600 if lat>=0 else 32700)+min(60,int((lon+180)//6)+1)
    crs=f'EPSG:{epsg}';project=Transformer.from_crs(4326,epsg,always_xy=True)
    projected=geom_transform(project.transform,polygon)
    if not 50_000<=projected.area<=150_000_000:raise ValueError('AOI_AREA_LIMIT')
    left,bottom,right,top=projected.bounds;res=20
    left=math.floor(left/res)*res;top=math.ceil(top/res)*res
    width=math.ceil((right-left)/res);height=math.ceil((top-bottom)/res)
    if width*height>500_000:raise ValueError('PIXEL_BUDGET_EXCEEDED')
    transform=from_origin(left,top,res,res)
    inside=geometry_mask([mapping(projected)],out_shape=(height,width),transform=transform,invert=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    audit={'schema':'BOUH_COVERAGE_RASTER_1','area_id':area_id,'started_at':stamp(),
           'processing_state':'STARTED','files':[],'scenes':[], 'errors':[],
           'aoi':area['geometry'],'grid':{'crs':crs,'resolution_m':20,'width':width,'height':height},
           'algorithm_sha256':sha(Path(__file__).with_name('bouh_math.py')),
           'collector_sha256':sha(__file__), 'gold_probability':None,'field_validated':False,
           'terrain_shadow_policy':'SCL_SHADOW_ONLY_NO_DEM_SHADOW_EXECUTED'}
    session=requests.Session();session.headers['User-Agent']='BOUH-Coverage-Collector/0.3.0'
    items,request=discover(session,area['geometry'],days,max_scenes)
    audit['catalog_request']=request
    write_json(out/'scenes.json',{'type':'FeatureCollection','features':items})
    profile=dict(driver='GTiff',height=height,width=width,count=1,dtype='float32',
                 crs=crs,transform=transform,nodata=-9999,compress='deflate')
    success=[]
    try:
        for number,item in enumerate(items):
            name=f'scene-{number:02d}';folder=out/name;folder.mkdir(exist_ok=True)
            entry={'product_id':item['id'],'acquisition_datetime':item['properties']['datetime'],
                   'state':'METADATA_INSPECTED','inputs':[]}
            audit['scenes'].append(entry)
            try:
                arrays={};scl=None
                for band in ['scl']+BANDS:
                    asset=item.get('assets',{}).get(band)
                    if not asset:raise ValueError('MISSING_BAND_'+band)
                    url=trusted_asset(asset.get('href'))
                    scale,offset=(1,0) if band=='scl' else calibration(asset)
                    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='45',
                                      GDAL_HTTP_MAX_RETRY='1',AWS_NO_SIGN_REQUEST='YES'):
                        with rasterio.open(url) as src:
                            if src.count!=1 or src.crs is None:raise ValueError('INVALID_SOURCE_RASTER')
                            native={'crs':str(src.crs),'resolution':list(src.res),'dtype':src.dtypes[0],
                                    'nodata':None if src.nodata is None or not math.isfinite(float(src.nodata)) else float(src.nodata)}
                            # Categorical SCL nearest; average native 10m SR to20m,
                            # retain20m SWIR support. Missingness remains a mask.
                            method=Resampling.nearest if band=='scl' else Resampling.average
                            with WarpedVRT(src,crs=crs,transform=transform,width=width,height=height,
                                           dtype='float32',nodata=float('nan'),resampling=method) as vrt:
                                data=vrt.read(1,masked=True).filled(np.nan)
                    data[~inside]=np.nan
                    if band=='scl':scl=data
                    else:arrays[band]=data*scale+offset
                    path=folder/(band+'.tif')
                    with rasterio.open(path,'w',**profile) as dst:dst.write(np.where(np.isfinite(data),data,-9999).astype('float32'),1)
                    # These are spatial/resampled DN derivatives, not full source bytes.
                    record={'file':str(path.relative_to(out)),'sha256':sha(path),'parent_product_id':item['id'],
                            'source_url':url,'parent_full_file_sha256':None,'source_native':native,
                            'scale':scale,'offset':offset,'state':'FILE_INSPECTED',
                            'artifact_type':'SPATIAL_RESAMPLED_DN_DERIVATIVE',
                            'resampling':'nearest' if band=='scl' else 'average',
                            'download_datetime':stamp()}
                    entry['inputs'].append(record);audit['files'].append(record)
                products=compute_indices(arrays,inside&(scl==5),np)
                n=int(np.isfinite(products['gossan_ratio']).sum());fraction=n/max(1,int(inside.sum()))
                entry.update(valid_fraction=fraction,state='PROCESSING_EXECUTED')
                if fraction<.10:entry['state']='LOW_VALID_COVERAGE';continue
                for layer,data in products.items():
                    if layer=='valid_bare_mask':
                        # QA0=invalid,1=eligible; NoData255 is outside AOI only.
                        pp=profile|{'dtype':'uint8','nodata':255}
                        data=np.where(inside,data,255).astype('uint8')
                    else:pp=profile;data=np.where(np.isfinite(data),data,-9999).astype('float32')
                    path=folder/(layer+'.tif')
                    with rasterio.open(path,'w',**pp) as dst:dst.write(data,1)
                    audit['files'].append({'file':str(path.relative_to(out)),'sha256':sha(path),
                        'state':'PROCESSING_EXECUTED','parent_product_id':item['id'],'artifact_type':'COMPUTED_SPECTRAL_PROXY'})
                success.append({'folder':name,'product_id':item['id'],'datetime':item['properties']['datetime']})
            except Exception as exc:
                entry.update(state='BLOCKED',reason=str(exc));audit['errors'].append({'product_id':item['id'],'error':str(exc)})
        if not success:raise ValueError('NO_SCENE_PASSED_RASTER_QA')
        # Latest QA-eligible scene, not a mosaic of unknown acquisition dates.
        audit['selected']=success[0];audit['processing_state']='PROCESSING_EXECUTED'
        return audit
    finally:
        audit['finished_at']=stamp();write_json(out/'manifest.json',audit)
        (out/'CHECKSUMS.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n'
             for p in sorted(out.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.sha256'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--area',default='focus-5km')
    parser.add_argument('--coverage',default='coverage-areas.geojson')
    parser.add_argument('--output',default='BOUH_COVERAGE_OUTPUT')
    args=parser.parse_args()
    collect(args.area,args.output,coverage_path=args.coverage)

"""BOUH acquisition pilot. Real outputs require live network + listed dependencies.
Catalogue != download != calibrated indicator != field-validated prospectivity.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import platform
import zipfile
from urllib.parse import urlparse
from bouh_math import calibration, compute_indices

REGION=[35.5,18.0,38.0,21.0]  # provisional envelope, not verified place boundaries
FOCUS=(18.8888,37.18893)
END=dt.datetime.now(dt.timezone.utc).date()
START=END.replace(year=END.year-1) if not(END.month==2 and END.day==29) else END.replace(year=END.year-1,day=28)
COLLECTIONS={
 's2_c1':('https://earth-search.aws.element84.com/v1','sentinel-2-c1-l2a'),
 's2_legacy':('https://earth-search.aws.element84.com/v1','sentinel-2-l2a'),
 'dem':('https://earth-search.aws.element84.com/v1','cop-dem-glo-30'),
 'landsat':('https://earth-search.aws.element84.com/v1','landsat-c2-l2'),
 's1':('https://stac.dataspace.copernicus.eu/v1','sentinel-1-grd')}

def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj): Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def safe_name(s): return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in s)[:160]
def https(url):
    if not isinstance(url,str) or urlparse(url).scheme!='https':raise ValueError('NON_HTTPS_ASSET')
    return url

def search_catalog(session,root,collection,bbox,start,end,max_items=300):
    payload={'collections':[collection],'bbox':bbox,'limit':100}
    if collection!='cop-dem-glo-30':payload.update(datetime=f'{start}T00:00:00Z/{end}T23:59:59Z',sortby=[{'field':'properties.datetime','direction':'desc'}])
    result=[];url=root+'/search';method='POST';truncated=False
    while url:
        if urlparse(url).hostname!=urlparse(root).hostname:raise ValueError('UNEXPECTED_PAGINATION_HOST')
        response=session.request(method,url,json=payload if method=='POST' else None,timeout=60)
        response.raise_for_status();page=response.json()
        if page.get('type')!='FeatureCollection' or not isinstance(page.get('features'),list):raise ValueError('INVALID_STAC_RESPONSE')
        result.extend(page['features']);next_link=next((l for l in page.get('links',[]) if l.get('rel')=='next'),None)
        if len(result)>=max_items:truncated=bool(next_link) or len(result)>max_items;result=result[:max_items];break
        if not next_link:break
        url=https(next_link['href']);method=next_link.get('method','GET').upper()
        if method not in ['GET','POST']:raise ValueError('INVALID_PAGINATION_METHOD')
        payload=next_link.get('body',payload)
    unique={f.get('id'):f for f in result if isinstance(f,dict) and f.get('type')=='Feature' and f.get('id')}
    return {'status':'METADATA_DOWNLOADED','items':list(unique.values()),'truncated':truncated,'observed_at':stamp(),'request':{'root':root,'collection':collection,'bbox':bbox,'start':str(start),'end':str(end)}}

def run(output='BOUH_DATA_PACKAGE',download_focus=True,max_scenes=3):
    import requests
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds
    from rasterio.transform import from_origin
    from rasterio.features import geometry_mask
    from pyproj import Geod,Transformer
    from shapely.geometry import shape,mapping
    from shapely.ops import transform as shape_transform
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']='BOUH-research-acquisition/1.0'
    geod=Geod(ellps='WGS84');lat,lon=FOCUS;points=[geod.fwd(lon,lat,i*360/180,5000)[:2] for i in range(180)];points.append(points[0]);focus={'type':'Polygon','coordinates':[points]};polygon=shape(focus)
    write_json(out/'AOI.geojson',{'type':'FeatureCollection','features':[{'type':'Feature','properties':{'radius_m':5000,'center':[lon,lat],'method':'WGS84 ellipsoidal geodesic'},'geometry':focus}]})
    audit={'schema':'BOUH_ACQUISITION_1','owner':'أحمد أبو عزيزه الرشيدي','started_at':stamp(),'regional_bbox':REGION,'region_boundaries_verified':False,'amur_location_verified':False,'software':{'python':platform.python_version(),'numpy':np.__version__,'rasterio':rasterio.__version__},'catalogues':{},'local_files':[],'processing':[],'errors':[],'core_score':None,'gold_probability':None}
    files=[];catalogues={}
    for key,(root,collection) in COLLECTIONS.items():
        try:
            # Regional discovery; download/processing remains limited to user focus.
            cat=search_catalog(session,root,collection,REGION,START,END)
            catalogues[key]=cat['items'];path=out/'metadata'/f'{key}.json';write_json(path,cat);files.append(path)
            audit['catalogues'][key]={'status':cat['status'],'count':len(cat['items']),'truncated':cat['truncated'],'metadata_file':str(path.relative_to(out))}
        except Exception as exc:audit['catalogues'][key]={'status':'BLOCKED_OR_FAILED','error':str(exc)[:500]}
        write_json(out/'run_audit.json',audit)
    project=Transformer.from_crs(4326,32637,always_xy=True);projected=shape_transform(project.transform,polygon);left,bottom,right,top=projected.bounds
    resolution=20;left=math.floor(left/resolution)*resolution;top=math.ceil(top/resolution)*resolution;width=math.ceil((right-left)/resolution);height=math.ceil((top-bottom)/resolution)
    if width*height>2_000_000:raise ValueError('PILOT_PIXEL_LIMIT_EXCEEDED')
    transform=from_origin(left,top,resolution,resolution);inside=geometry_mask([mapping(projected)],out_shape=(height,width),transform=transform,invert=True)
    script_sha=digest(__file__)
    def read_asset(item,key,folder,categorical=False):
        asset=item.get('assets',{}).get(key)
        if not asset:raise ValueError('MISSING_ASSET_'+key)
        url=https(asset.get('href'));path=folder/(safe_name(key)+'.tif');path.parent.mkdir(parents=True,exist_ok=True)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='60'):
            with rasterio.open(url) as src:
                native={'crs':str(src.crs),'transform':list(src.transform),'width':src.width,'height':src.height,'dtype':src.dtypes[0],'nodata':None if src.nodata is None or not math.isfinite(float(src.nodata)) else float(src.nodata),'scales':list(src.scales),'offsets':list(src.offsets)}
                if src.count!=1:raise ValueError('EXPECTED_SINGLE_BAND_'+key)
                with WarpedVRT(src,crs='EPSG:32637',transform=transform,width=width,height=height,resampling=Resampling.nearest if categorical else Resampling.bilinear,dtype='float32',nodata=float('nan')) as vrt:data=vrt.read(1,masked=True).filled(np.nan).astype('float32')
        data[~inside]=np.nan
        with rasterio.open(path,'w',driver='GTiff',height=height,width=width,count=1,dtype='float32',crs='EPSG:32637',transform=transform,nodata=float('nan'),compress='deflate') as dst:dst.write(data,1)
        entry={'file':str(path.relative_to(out)),'sha256':digest(path),'parent_remote_url':url,'parent_product_id':item['id'],'parent_full_file_sha256':None,'status':'WINDOWED_REPROJECTED_DN_DERIVATIVE','source_native':native,'processing_script_sha256':script_sha,'target_grid_m':resolution,'resampling':'nearest' if categorical else 'bilinear','remote_file_fully_downloaded':False}
        audit['local_files'].append(entry);files.append(path);return data,asset,entry
    s2=[]
    for key in ['s2_c1','s2_legacy']:
        for item in catalogues.get(key,[]):
            try:
                if shape(item['geometry']).intersection(polygon).area>0 and item['properties'].get('eo:cloud_cover',100)<=20:s2.append(item)
            except Exception:continue
    # Keep distinct products, prefer lower scene cloud, then newer observation. Not a scene-level quality proof.
    s2.sort(key=lambda x:str(x['properties'].get('datetime','')),reverse=True);s2.sort(key=lambda x:x['properties'].get('eo:cloud_cover',100))
    seen=set();selected=[]
    for item in s2:
        product=item['properties'].get('s2:product_uri',item['id'])
        if product in seen:continue
        seen.add(product);selected.append(item)
        if len(selected)>=max_scenes:break
    audit['focus_selected_s2_ids']=[x['id'] for x in selected]
    if download_focus:
        for item in selected:
            folder=out/'inputs'/safe_name(item['id'])
            try:
                scl,_,scl_entry=read_asset(item,'scl',folder,True);valid=inside & (scl==5)  # conservative bare land only
                arrays={};provenance={}
                for key in ['blue','red','nir','swir16','swir22']:
                    dn,asset,entry=read_asset(item,key,folder);scale,offset=calibration(asset);arrays[key]=dn*scale+offset;provenance[key]={**entry,'scale':scale,'offset':offset}
                outputs=compute_indices(arrays,valid,np);processed=[]
                for key,data in outputs.items():
                    path=out/'derived'/safe_name(item['id'])/(key+'.tif');path.parent.mkdir(parents=True,exist_ok=True)
                    dtype='uint8' if key=='valid_bare_mask' else 'float32';nodata=0 if dtype=='uint8' else float('nan')
                    with rasterio.open(path,'w',driver='GTiff',height=height,width=width,count=1,dtype=dtype,crs='EPSG:32637',transform=transform,nodata=nodata,compress='deflate') as dst:dst.write(data.astype(dtype),1)
                    files.append(path);processed.append({'file':str(path.relative_to(out)),'sha256':digest(path)})
                denominator=int(inside.sum());valid_n=int(outputs['valid_bare_mask'].sum())
                audit['processing'].append({'product_id':item['id'],'status':'COMPUTED_REMOTE_INDICATORS','grid_resolution_m':20,'native_detail_not_increased':True,'qa_rule':'SCL==5, positive reflectance, stable denominators, NDVI<0.25','scl_input':scl_entry,'inputs':provenance,'inside_pixels':denominator,'valid_bare_pixels':valid_n,'valid_fraction':valid_n/denominator if denominator else None,'outputs':processed,'indicator_definitions':{'swir_ratio':{'formula':'SR_B11/SR_B12','evidence_family':'SPECTRAL_SWIR_ALTERATION','source_dependency_group':item['id']},'iron_ratio':{'formula':'SR_B04/SR_B02','evidence_family':'SPECTRAL_VNIR_IRON','source_dependency_group':item['id']}},'processing_version':'BOUH_RAW_PILOT_1','processing_script_sha256':script_sha,'field_validation':'UNVALIDATED','core_score':None})
            except Exception as exc:audit['errors'].append({'product_id':item.get('id'),'stage':'S2_FOCUS_PROCESSING','error':str(exc)[:700]})
            write_json(out/'run_audit.json',audit)
        # Download the DEM windows separately at native grid; no drainage or sink-filling claim.
        for item in catalogues.get('dem',[]):
            try:
                if shape(item['geometry']).intersection(polygon).area<=0:continue
                asset=item.get('assets',{}).get('data') or item.get('assets',{}).get('dem')
                if not asset:raise ValueError('MISSING_DEM_ASSET')
                url=https(asset.get('href'));path=out/'dem'/f'{safe_name(item["id"])}.tif';path.parent.mkdir(parents=True,exist_ok=True)
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='60'):
                    with rasterio.open(url) as src:
                        from rasterio.windows import from_bounds,Window
                        bounds=transform_bounds(4326,src.crs,*polygon.bounds,densify_pts=21)
                        window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths().intersection(Window(0,0,src.width,src.height));data=src.read(window=window);profile=src.profile.copy();profile.update(width=data.shape[2],height=data.shape[1],transform=src.window_transform(window),driver='GTiff',compress='deflate')
                        with rasterio.open(path,'w',**profile) as dst:dst.write(data)
                files.append(path);audit['local_files'].append({'file':str(path.relative_to(out)),'sha256':digest(path),'parent_remote_url':url,'parent_product_id':item['id'],'parent_full_file_sha256':None,'status':'NATIVE_GRID_DEM_WINDOW','hydrological_processing_executed':False})
            except Exception as exc:audit['errors'].append({'product_id':item.get('id'),'stage':'DEM_FOCUS_DOWNLOAD','error':str(exc)[:700]})
    audit['finished_at']=stamp();audit['field_validation_status']='UNVALIDATED';audit['local_raster_files']=sum(p.suffix=='.tif' for p in files);audit['processing_executed']=bool(audit['processing']);audit['regional_raster_coverage_verified']=False
    write_json(out/'run_audit.json',audit)
    checks=[]
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name!='CHECKSUMS.sha256':checks.append(f'{digest(path)}  {path.relative_to(out).as_posix()}')
    (out/'CHECKSUMS.sha256').write_text('\n'.join(checks)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    print(json.dumps({'archive':str(archive),'catalogues':audit['catalogues'],'rasters':audit['local_raster_files'],'processed_scenes':len(audit['processing']),'errors':len(audit['errors']),'core_score':None},ensure_ascii=False,indent=2))
    return audit,archive

if __name__=='__main__':run()

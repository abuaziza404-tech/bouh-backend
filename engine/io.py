"""STAC discovery and COG window reads. Rasterio is imported only by workers."""
from datetime import datetime,timedelta,timezone
import math
from urllib.parse import urlparse
import numpy as np

ROOT='https://earth-search.aws.element84.com/v1'
S2='sentinel-2-c1-l2a'
DEM='cop-dem-glo-30'


def search(session,collection,geometry,lookback_days=30,maximum=200):
    end=datetime.now(timezone.utc);start=end-timedelta(days=lookback_days)
    body={'collections':[collection],'intersects':geometry,'limit':100}
    if collection==S2:
        body.update(datetime=start.isoformat()+'/'+end.isoformat(),query={'eo:cloud_cover':{'lte':20}})
    url=ROOT+'/search';method='POST';items=[];seen=set()
    while url:
        signature=(url,repr(body) if method=='POST' else '')
        if signature in seen:raise ValueError('STAC_PAGINATION_LOOP')
        seen.add(signature)
        u=urlparse(url)
        if u.scheme!='https' or u.hostname!='earth-search.aws.element84.com':raise ValueError('UNTRUSTED_STAC_LINK')
        response=session.request(method,url,json=body if method=='POST' else None,timeout=(10,40));response.raise_for_status()
        data=response.json()
        if data.get('type')!='FeatureCollection' or not isinstance(data.get('features'),list):raise ValueError('INVALID_STAC_RESPONSE')
        items.extend(data['features'])
        next_link=next((x for x in data.get('links',[]) if x.get('rel')=='next'),None)
        if len(items)>maximum or (next_link and len(items)>=maximum):raise ValueError('TOO_MANY_SCENES_NARROW_LOOKBACK')
        if not next_link:break
        url=next_link['href'];method=next_link.get('method','GET').upper()
        if method not in {'GET','POST'}:raise ValueError('INVALID_STAC_METHOD')
        body=next_link.get('body',body)
    unique={x['id']:x for x in items if x.get('type')=='Feature' and x.get('id')}
    return sorted(unique.values(),key=lambda f:f.get('properties',{}).get('datetime',''),reverse=True)


def asset_url(url,dem=False):
    u=urlparse(url or '');host=u.hostname or ''
    allowed=(host=='copernicus-dem-30m.s3.eu-central-1.amazonaws.com' if dem
             else host.endswith('.s3.us-west-2.amazonaws.com'))
    if not allowed or u.scheme!='https' or u.username or u.password or u.query or u.fragment or u.port not in (None,443) or not u.path.lower().endswith('.tif'):
        raise ValueError('UNTRUSTED_COG_ASSET')
    return '/vsicurl/'+url


def calibration(asset):
    r=asset.get('raster:bands',[])
    if len(r)!=1 or 'scale' not in r[0] or 'offset' not in r[0]:raise ValueError('MISSING_SCALE_OFFSET')
    scale,offset=r[0]['scale'],r[0]['offset']
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in [scale,offset]) or scale<=0:raise ValueError('INVALID_SCALE_OFFSET')
    return scale,offset


def read_cog(asset,crs,transform,width,height,kind='reflectance'):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    url=asset_url(asset['href'],dem=(kind=='dem'))
    method={'reflectance':Resampling.average,'qa':Resampling.nearest,'dem':Resampling.bilinear}[kind]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',
                      GDAL_HTTP_MAX_RETRY='1',GDAL_CACHEMAX=32*1024*1024,AWS_NO_SIGN_REQUEST='YES'):
        with rasterio.open(url) as src:
            if src.count!=1 or src.crs is None:raise ValueError('INVALID_RASTER_STRUCTURE')
            with WarpedVRT(src,crs=crs,transform=transform,width=width,height=height,
                           dtype='float32',nodata=float('nan'),resampling=method) as vrt:
                data=vrt.read(1,masked=True).filled(np.nan)
            metadata={'source_url':asset['href'],'source_crs':str(src.crs),'source_resolution':list(src.res),
                      'dtype':src.dtypes[0],'parent_full_file_sha256':None,
                      'access':'COG_WINDOW_READ_NOT_FULL_DOWNLOAD','resampling':method.name}
    return data,metadata


def solar_geometry(item):
    p=item.get('properties',{})
    az=p.get('view:sun_azimuth',p.get('s2:mean_solar_azimuth'))
    elevation=p.get('view:sun_elevation')
    if elevation is None and p.get('s2:mean_solar_zenith') is not None:elevation=90-p['s2:mean_solar_zenith']
    if az is None or elevation is None or not (0<=az<=360 and 15<=elevation<90):raise ValueError('MISSING_OR_LOW_SOLAR_GEOMETRY')
    return float(az),float(elevation)

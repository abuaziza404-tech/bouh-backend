"""Server-owned coverage areas and fresh background raster jobs."""
import os
import re
from pathlib import Path
from fastapi import APIRouter,Request,Depends,HTTPException,Header
from fastapi.responses import FileResponse
from .store import AdmissionError,IdempotencyConflict
from acquisition.coverage import load_areas,require_processable


def attach_coverage(app,owner,read_job,public_job):
    router=APIRouter(prefix='/api/v1')
    def areas():return load_areas(os.getenv('BOUH_COVERAGE_FILE','coverage-areas.geojson'))

    @router.get('/coverage')
    def coverage(user=Depends(owner)):
        return {'type':'FeatureCollection','features':[{'type':'Feature','id':a['id'],
          'properties':a['properties'],'geometry':a['geometry']} for a in areas().values()]}

    @router.post('/coverage/{area_id}/refresh',status_code=202)
    def refresh(area_id:str,request:Request,idempotency_key:str=Header(alias='Idempotency-Key'),user=Depends(owner)):
        if not re.fullmatch('[A-Za-z0-9_-]{16,80}',idempotency_key):raise HTTPException(422,'Invalid Idempotency-Key')
        area=areas().get(area_id)
        if not area:raise HTTPException(404,'Unknown coverage area')
        try:require_processable(area)
        except ValueError as exc:raise HTTPException(422,str(exc))
        payload={'_processor':'coverage','area_id':area_id,'days':30,'max_scenes':3}
        try:job=request.app.state.store.create(user,idempotency_key,payload)
        except AdmissionError:raise HTTPException(429,'Queue full',headers={'Retry-After':'30'})
        except IdempotencyConflict:raise HTTPException(409,'Idempotency conflict')
        return public_job(job)

    @router.get('/jobs/{ident}/overlay/{layer}.png')
    def image(ident:str,layer:str,request:Request,user=Depends(owner)):
        job=read_job(request,ident,user)
        if job['state']!='completed':raise HTTPException(409,'Not ready')
        if layer not in {'iron_ratio','swir_ratio','gossan_ratio'}:raise HTTPException(404,'Unknown layer')
        p=request.app.state.store.directory/('rasters-'+ident)/(layer+'.png')
        if not p.is_file():raise HTTPException(404,'Layer unavailable')
        return FileResponse(p,media_type='image/png',headers={'Cache-Control':'private, no-store'})
    @router.get('/jobs/{ident}/raster/{layer}.tif')
    def raster(ident:str,layer:str,request:Request,user=Depends(owner)):
        import json
        job=read_job(request,ident,user)
        if job['state']!='completed':raise HTTPException(409,'Not ready')
        if layer not in {'iron_ratio','swir_ratio','gossan_ratio','ndvi','ndwi','valid_bare_mask'}:
            raise HTTPException(404,'Unknown layer')
        root=request.app.state.store.directory/('rasters-'+ident)
        manifest=root/'manifest.json'
        if not manifest.is_file():raise HTTPException(404,'No raster manifest')
        folder=json.loads(manifest.read_text()).get('selected',{}).get('folder','')
        if not re.fullmatch(r'scene-[0-9]{2}',folder):raise HTTPException(404,'Invalid scene')
        p=root/folder/(layer+'.tif')
        if not p.is_file():raise HTTPException(404,'No raster')
        return FileResponse(p,media_type='image/tiff',filename=ident+'-'+layer+'.tif')
    app.include_router(router)


def process_coverage(payload,result_path):
    import json
    from acquisition.collect_bouh import collect,sha,stamp
    from acquisition.overlays import create_overlay
    path=Path(result_path);job_id=path.stem
    root=path.parent/('rasters-'+job_id)
    audit=collect(payload['area_id'],root,payload['days'],payload['max_scenes'],
                  os.getenv('BOUH_COVERAGE_FILE','coverage-areas.geojson'))
    folder=root/audit['selected']['folder'];overlays=[]
    for key,color in [('iron_ratio',(228,82,45)),('swir_ratio',(22,166,182)),('gossan_ratio',(205,140,50))]:
        entry=create_overlay(folder/(key+'.tif'),root/(key+'.png'),color)
        entry.update(layer=key,url=f'/api/v1/jobs/{job_id}/overlay/{key}.png',
                     sha256=sha(root/(key+'.png')),raster_url=f'/api/v1/jobs/{job_id}/raster/{key}.tif',source_product_id=audit['selected']['product_id'])
        overlays.append(entry)
    body={'type':'FeatureCollection','features':[],
          'metadata':{'schema_version':'BOUH_RASTER_OVERLAYS_1','area_id':payload['area_id'],
          'received_at':stamp(),'acquisition_datetime':audit['selected']['datetime'],
          'valid_fraction':next(s['valid_fraction'] for s in audit['scenes'] if s['product_id']==audit['selected']['product_id']),
          'overlays':overlays,'field_validated':False,'gold_probability':None,
          'qa':'SCL5 + finite reflectance + stable denominators + NDVI/NDWI',
          'terrain_shadow_policy':'SCL_ONLY_NO_DEM_SHADOW_EXECUTED',
          'state':'PROCESSING_EXECUTED','note':'Ratios only; not the Crosta/DEM evidence pipeline'}}
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(body,allow_nan=False));tmp.replace(path)

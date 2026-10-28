"""Single-user FastAPI engine, suitable for a bounded free-tier trial."""
from contextlib import asynccontextmanager
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
import uuid
from fastapi import FastAPI,Depends,HTTPException,Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer,HTTPAuthorizationCredentials
from engine.coverage import load_areas,require_processable
from engine.science import xyz_bounds

COVERAGE=os.getenv('COVERAGE_FILE','coverage-areas.geojson')
CACHE=Path(os.getenv('CACHE_DIR','/tmp/bouh-engine'))
STATE={};LOCK=threading.Lock();SLOT=threading.BoundedSemaphore(1)
TILE_SLOTS=threading.BoundedSemaphore(3)
AUTH=HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app):
    if len(os.getenv('BOUH_API_TOKEN',''))<32:raise RuntimeError('Set a random BOUH_API_TOKEN of at least32 characters')
    load_areas(COVERAGE);CACHE.mkdir(parents=True,exist_ok=True)
    yield

app=FastAPI(title='BOUH Terrain Tile Engine',version='0.4.0',lifespan=lifespan)
origins=[v.strip() for v in os.getenv('CORS_ORIGINS','null').split(',') if v.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Authorization'])


def owner(credential:HTTPAuthorizationCredentials|None=Depends(AUTH)):
    token=os.getenv('BOUH_API_TOKEN','')
    if not credential or credential.scheme.lower()!='bearer' or not secrets.compare_digest(credential.credentials,token):
        raise HTTPException(401,'Authentication required',headers={'WWW-Authenticate':'Bearer'})


def area_config(area_id):
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}',area_id):raise HTTPException(404,'Area not found')
    area=load_areas(COVERAGE).get(area_id)
    if area is None:raise HTTPException(404,'Area not found')
    return area


def child(area_id,path):
    from engine.process import execute
    execute(area_id,path,COVERAGE)


def run(area_id,revision,out):
    proc=multiprocessing.get_context('spawn').Process(target=child,args=(area_id,str(out)))
    try:
        proc.start();proc.join(timeout=int(os.getenv('JOB_TIMEOUT_SECONDS','600')))
        if proc.is_alive():
            proc.terminate();proc.join(timeout=5)
            if proc.is_alive():proc.kill();proc.join(timeout=2)
            raise RuntimeError('PROCESSING_TIMEOUT')
        if proc.exitcode!=0:raise RuntimeError('SOURCE_OR_QA_OR_PCA_GATE_FAILED')
        manifest=json.loads((out/'manifest.json').read_text())
        manifest['revision']=revision
        with LOCK:
            STATE[area_id]={'state':'completed','revision':revision,'out':str(out),
                            'completed_at':time.time(),'manifest':manifest}
    except Exception as exc:
        with LOCK:STATE[area_id]={'state':'failed','revision':revision,'error':str(exc)}
    finally:SLOT.release()


@app.get('/healthz')
def health():return {'status':'ok','cache':'ephemeral','probabilities':'not_calibrated'}


@app.get('/api/coverage')
def coverage(user=Depends(owner)):
    return {'type':'FeatureCollection','features':[{'type':'Feature','id':a['id'],
            'properties':a['properties'],'geometry':a['geometry']} for a in load_areas(COVERAGE).values()]}


@app.post('/api/coverage/{area_id}/refresh',status_code=202)
def refresh(area_id:str,force:bool=False,user=Depends(owner)):
    area=area_config(area_id)
    try:require_processable(area)
    except ValueError as exc:raise HTTPException(422,str(exc))
    with LOCK:
        prior=STATE.get(area_id)
        if prior and prior['state']=='processing':return {'state':'processing','revision':prior['revision']}
        if prior and prior['state']=='completed' and not force and time.time()-prior['completed_at']<6*3600:
            return {'state':'completed','revision':prior['revision'],'cache_reused':True}
        if not SLOT.acquire(blocking=False):raise HTTPException(429,'One job is already running',headers={'Retry-After':'20'})
        # Limit retained local results; free instances have no durable disk.
        for ident in list(STATE):
            if ident!=area_id and STATE[ident].get('out'):
                shutil.rmtree(STATE[ident]['out'],ignore_errors=True);del STATE[ident]
        if prior and prior.get('out'):shutil.rmtree(prior['out'],ignore_errors=True)
        rev=uuid.uuid4().hex;out=CACHE/(area_id+'-'+rev)
        try:out.mkdir(parents=True)
        except Exception:
            SLOT.release()
            raise HTTPException(503,'Cannot allocate processing workspace')
        STATE[area_id]={'state':'processing','revision':rev,'started_at':time.time()}
    try:threading.Thread(target=run,args=(area_id,rev,out),daemon=True).start()
    except Exception:
        with LOCK:STATE[area_id]={'state':'failed','revision':rev,'error':'WORKER_START_FAILED'}
        SLOT.release()
        raise HTTPException(503,'Cannot start worker')
    return {'state':'processing','revision':rev}


@app.get('/api/coverage/{area_id}/status')
def status(area_id:str,user=Depends(owner)):
    area_config(area_id)
    with LOCK:state=STATE.get(area_id,{'state':'not_processed'}).copy()
    state.pop('out',None)
    if state['state']=='completed':
        state['tile_template']=f"/api/coverage/{area_id}/tiles/{{z}}/{{x}}/{{y}}.png?v={state['revision']}"
    return state


@lru_cache(maxsize=64)
def cached_tile(path,z,x,y):
    from engine.tiles import tile_png
    return tile_png(path,z,x,y)


@app.get('/api/coverage/{area_id}/tiles/{z}/{x}/{y}.png')
def tile(area_id:str,z:int,x:int,y:int,v:str|None=None,user=Depends(owner)):
    area_config(area_id)
    try:xyz_bounds(z,x,y)
    except ValueError:raise HTTPException(422,'Invalid XYZ coordinates')
    with LOCK:state=STATE.get(area_id,{}).copy()
    if state.get('state')!='completed':raise HTTPException(409,'Prepare the area with POST refresh first')
    if v is not None and v!=state['revision']:raise HTTPException(409,'Layer revision changed; refresh status')
    if not TILE_SLOTS.acquire(blocking=False):raise HTTPException(503,'Tile capacity busy',headers={'Retry-After':'2'})
    try:
        body=cached_tile(str(Path(state['out'])/'classes.tif'),z,x,y)
        return Response(body,media_type='image/png',headers={'Cache-Control':'private,max-age=600'})
    except Exception:raise HTTPException(503,'Tile temporarily unavailable')
    finally:TILE_SLOTS.release()

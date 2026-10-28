"""FastAPI: short HTTP requests + durable background jobs + bounded GeoJSON."""
from contextlib import asynccontextmanager
import hashlib
import json
import os
import secrets
import re
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError
from .models import AnalysisRequest
from .store import JobStore, AdmissionError, IdempotencyConflict
from .worker import Worker


@asynccontextmanager
async def lifespan(app):
    token=os.environ.get('BOUH_API_TOKEN','')
    if len(token)<32: raise RuntimeError('Configure BOUH_API_TOKEN with >=32 random characters')
    if os.getenv('BOUH_PROCESSOR','coverage')=='gee' and not os.environ.get('GEE_PROJECT'): raise RuntimeError('Configure GEE_PROJECT')
    store=JobStore(os.environ.get('BOUH_DATA_DIR','/tmp/bouh-jobs'))
    worker=Worker(store); app.state.store=store; worker.start()
    from terrain_api import app as terrain_app, lifespan as terrain_lifespan
    try:
        async with terrain_lifespan(terrain_app):
            yield
    finally: worker.close()


app=FastAPI(title='BOUH Terrain Cloud Bridge',version='0.2.0',lifespan=lifespan)
origins=[x.strip() for x in os.getenv('BOUH_CORS_ORIGINS','').split(',') if x.strip()]
if origins:
    app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],
                       allow_headers=['Authorization','Content-Type','Idempotency-Key'])
app.add_middleware(GZipMiddleware,minimum_size=1000)
bearer=HTTPBearer(auto_error=False)


def owner(credentials: HTTPAuthorizationCredentials | None=Depends(bearer)):
    expected=os.environ.get('BOUH_API_TOKEN','')
    if not credentials or credentials.scheme.lower()!='bearer' or not secrets.compare_digest(credentials.credentials,expected):
        raise HTTPException(401,'Authentication required',headers={'WWW-Authenticate':'Bearer'})
    # Personal single-user deployment. Replace this with OIDC/JWT per-user
    # identity before sharing publicly. Never embed a server token in an APK.
    return hashlib.sha256(credentials.credentials.encode()).hexdigest()


def read_job(request,ident,user):
    if not re.fullmatch('[0-9a-f]{32}',ident): raise HTTPException(404,'Job not found')
    row=request.app.state.store.get(ident,user)
    if not row: raise HTTPException(404,'Job not found')
    return row


def public_job(row):
    return {'job_id':row['id'],'state':row['state'],'error_code':row['error'],
            'status_url':f"/api/v1/jobs/{row['id']}",
            'result_url':f"/api/v1/jobs/{row['id']}/result",'poll_after_seconds':5}


@app.get('/healthz')
def health():
    return {'status':'ok','mode':'single-instance-durable-worker'}


@app.post('/api/v1/analyze-terrain',status_code=202,openapi_extra={
    'requestBody':{'required':True,'content':{'application/json':{'schema':AnalysisRequest.model_json_schema()}}}})
async def analyze(request:Request,user=Depends(owner),idempotency_key:str=Header(alias="Idempotency-Key")):
    # Streaming size cap works even without Content-Length/chunked uploads.
    raw=bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>8192: raise HTTPException(413,'Request too large')
    try:
        payload=AnalysisRequest.model_validate_json(bytes(raw))
    except ValidationError as exc:
        raise HTTPException(422,exc.errors(include_url=False,include_context=False,include_input=False))
    idem=idempotency_key
    if not re.fullmatch('[A-Za-z0-9_-]{16,80}',idem):
        raise HTTPException(422,'Idempotency-Key must contain 16-80 letters/digits/_/-')
    try:
        # Persist canonical bbox, not duplicate polygon+bbox; model accepts one.
        data=payload.model_dump(mode='json',exclude={'polygon'})
        row=request.app.state.store.create(user,idem,data)
    except AdmissionError:
        raise HTTPException(429,'Queue full',headers={'Retry-After':'30'})
    except IdempotencyConflict:
        raise HTTPException(409,'Idempotency-Key was already used for a different request')
    return public_job(row)


@app.get('/api/v1/jobs/{ident}')
def status(ident:str,request:Request,user=Depends(owner)):
    return public_job(read_job(request,ident,user))


@app.get('/api/v1/jobs/{ident}/result')
def result(ident:str,request:Request,user=Depends(owner)):
    row=read_job(request,ident,user)
    if row['state']=='failed': raise HTTPException(422,row['error'])
    if row['state']!='completed': raise HTTPException(409,'Result not ready')
    path=request.app.state.store.result_path(ident)
    if not path.exists(): raise HTTPException(410,'Result is unavailable')
    return FileResponse(path,media_type='application/geo+json',headers={'Cache-Control':'private, no-store'})

from .coverage_routes import attach_coverage
attach_coverage(app,owner,read_job,public_job)

# Both engines share one HTTPS origin and owner token.
from terrain_api import app as terrain_app
app.mount("/terrain", terrain_app)

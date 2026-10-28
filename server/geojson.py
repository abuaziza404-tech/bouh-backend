"""Compact mobile response, with hard limits and preserved polygon holes."""
import json
import math

MAX_FEATURES=300
MAX_VERTICES=40000
MAX_BYTES=2_000_000


def compact_collection(collection, metadata):
    if collection.get('type')!='FeatureCollection': raise ValueError('Invalid collection')
    features=collection.get('features',[])
    if len(features)>MAX_FEATURES: raise ValueError('TOO_MANY_FEATURES_USE_SMALLER_AOI')
    count=0
    def coordinate(p):
        nonlocal count
        count+=1
        if count>MAX_VERTICES: raise ValueError('TOO_MANY_VERTICES_USE_SMALLER_AOI')
        if len(p)<2 or any(not math.isfinite(float(x)) for x in p[:2]):
            raise ValueError('Invalid coordinate')
        x,y=map(float,p[:2])
        if not (-180<=x<=180 and -90<=y<=90): raise ValueError('Coordinate out of range')
        return [round(x,6),round(y,6)]
    out=[]
    for i,f in enumerate(features):
        g=f.get('geometry') or {}; typ=g.get('type'); coords=g.get('coordinates',[])
        if typ not in ('Polygon','MultiPolygon'): raise ValueError('Expected polygons')
        polygons=[coords] if typ=='Polygon' else coords
        clean=[]
        for poly in polygons:
            rings=[]
            for ring in poly:
                r=[coordinate(p) for p in ring]
                if len(r)<4 or r[0]!=r[-1] or len({tuple(x) for x in r[:-1]})<3:
                    raise ValueError('Invalid or precision-collapsed ring')
                rings.append(r)
            if not rings: raise ValueError('Empty polygon')
            clean.append(rings)
        p=f.get('properties',{})
        props={k:p[k] for k in ('evidence_kind','evidence_tier','mean_persistence',
               'mean_valid_dates','area_m2','classification','target_id') if k in p}
        out.append({'type':'Feature','id':str(i),'properties':props,
                    'geometry':{'type':typ,'coordinates':clean[0] if typ=='Polygon' else clean}})
    result={'type':'FeatureCollection','features':out,'metadata':metadata}
    body=json.dumps(result,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    if len(body.encode())>MAX_BYTES: raise ValueError('RESULT_TOO_LARGE_USE_SMALLER_AOI')
    return body

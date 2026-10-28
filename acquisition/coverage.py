"""Server-owned AOIs, validated before admitting any paid/network work."""
import json
import math
import re
from pathlib import Path


def load_areas(path):
    raw=json.loads(Path(path).read_text())
    if raw.get('type')!='FeatureCollection':raise ValueError('INVALID_COVERAGE_FILE')
    result={}
    for f in raw.get('features',[]):
        ident=f.get('id');g=f.get('geometry',{});p=f.get('properties',{})
        if not isinstance(ident,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,64}',ident) or ident in result:
            raise ValueError('MISSING_OR_DUPLICATE_AREA_ID')
        if g.get('type')!='Polygon' or len(g.get('coordinates',[]))!=1:
            raise ValueError('SIMPLE_POLYGON_REQUIRED')
        ring=g['coordinates'][0]
        if not 4<=len(ring)<=2000 or ring[0]!=ring[-1]:raise ValueError('INVALID_RING')
        for q in ring:
            if len(q)!=2 or any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in q):
                raise ValueError('INVALID_COORDINATES')
            if not (-180<=q[0]<=180 and -60<=q[1]<=75):raise ValueError('OUT_OF_RANGE')
        xs,ys=zip(*ring);bbox=[min(xs),min(ys),max(xs),max(ys)]
        if bbox[2]-bbox[0]>=180:raise ValueError('ANTIMERIDIAN_NOT_SUPPORTED')
        result[ident]={'id':ident,'geometry':g,'properties':p,'bbox':bbox}
    if not result:raise ValueError('NO_AREAS')
    return result


def require_processable(area):
    if area['properties'].get('processing_enabled') is not True:
        raise ValueError('AREA_DISABLED_CONTEXT_ONLY')
    w,s,e,n=area['bbox']
    estimate=(e-w)*111.32*math.cos(math.radians((s+n)/2))*(n-s)*111.32
    if not .05<=estimate<=150 or e-w>.2 or n-s>.2:
        raise ValueError('AREA_REQUIRES_SMALLER_PROCESSING_TILES')
    return area

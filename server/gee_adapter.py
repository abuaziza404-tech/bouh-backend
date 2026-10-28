"""Executed in an isolated worker process, never in the HTTP event loop."""
import json
import os
from pathlib import Path
from .geojson import compact_collection, MAX_FEATURES


def run_analysis(payload, result_path):
    import ee
    import google.auth
    from bouh.config import Config
    from bouh.pipeline import process
    from .models import AnalysisRequest
    request=AnalysisRequest.model_validate(payload)
    project=os.environ['GEE_PROJECT']
    credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/earthengine',
                                             'https://www.googleapis.com/auth/cloud-platform'])
    ee.Initialize(credentials=credentials,project=project)
    w,s,e,n=request.bbox
    zone=min(60,int(((w+e)/2+180)//6)+1)
    crs=f"EPSG:{32600+zone if (s+n)/2>=0 else 32700+zone}"
    cfg=Config(project=project,aoi_geojson=request.aoi(),start=str(request.start_date),
               end=str(request.end_date),crs=crs,max_scenes=40,
               reviewed_structures_asset=os.getenv('BOUH_STRUCTURES_ASSET') or None,
               redbeds_asset=os.getenv('BOUH_REDBEDS_ASSET') or None,
               geology_coverage_asset=os.getenv('BOUH_GEOLOGY_COVERAGE_ASSET') or None,
               geology_reference=os.getenv('BOUH_GEOLOGY_REFERENCE') or None)
    report={}
    try:
        product=process(cfg, report=report)
        targets=product.targets.filter(ee.Filter.gte('evidence_tier',request.min_tier))
        targets=targets.map(lambda f: ee.Feature(f).set('evidence_kind','intersection'))
        # Separate marginal PC evidence is display context, NOT additional
        # independent support or 'high confidence'. Keep it optional and tier1.
        if request.include_proxy_layers:
            for kind in ('ferric','hydroxyl'):
                p=product.evidence.select(kind+'_persistence')
                valid=product.evidence.select(kind+'_valid_date_count')
                mask=p.gte(cfg.min_persistence).And(valid.gte(cfg.min_observations))
                mask=mask.And(product.evidence.select('redbed_excluded').Not())
                mask=mask.And(mask.selfMask().connectedPixelCount(256,True).gte(cfg.min_component_pixels))
                fc=mask.toInt().selfMask().rename('class').reduceToVectors(
                    reducer=ee.Reducer.countEvery(),geometry=product.region,
                    crs=cfg.crs,scale=cfg.scale,geometryType='polygon',
                    maxPixels=cfg.max_pixels,tileScale=cfg.tile_scale)
                def decorate(f, kind=kind):
                    return ee.Feature(f).set({'evidence_kind':kind,'evidence_tier':1,
                        'classification':'SPECTRAL_CONTEXT_NOT_VALIDATED'})
                targets=targets.merge(fc.map(decorate))
        # Simplify on EE server before transfer. Do not truncate scientific
        # results silently: an oversized result fails with a smaller-AOI hint.
        targets=targets.map(lambda f: ee.Feature(f).setGeometry(ee.Feature(f).geometry().simplify(5)))
        raw=targets.limit(MAX_FEATURES+1).getInfo()
        metadata={'schema_version':'bouh-mobile-1','analysis_scale_m':20,
                  'geometry_simplification_m':5,'coordinate_rounding_decimals':6,
                  'field_validated':False,'gold_probability':None,
                  'config_sha256':product.report['config_sha256'],
                  'warnings':product.report['warnings'],
                  'min_intersection_tier':request.min_tier,
                  'proxy_layers_are_context':True}
        body=compact_collection(raw,metadata)
        path=Path(result_path); tmp=path.with_suffix('.tmp')
        tmp.write_text(body); tmp.replace(path)
    finally:
        # Small report survives a scientific gate failure. No secret credentials.
        Path(result_path).with_suffix('.manifest.json').write_text(
            json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))

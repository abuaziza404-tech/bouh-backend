"""Asynchronous, explicit export lifecycle; no invented completed downloads."""
import ee
from .numeric import QualityGateError

NODATA = -9999


def build_exports(product, cfg):
    prefix = "BOUH_"+product.report["config_sha256"][:12]
    # Float GeoTIFF separates NoData from valid zero tiers/counts. Masks for
    # numerical evidence and categorical QA remain separate named bands.
    raster = product.evidence.toFloat().unmask(NODATA, sameFootprint=False).clip(product.region)
    jobs = {
        "evidence_geotiff": ee.batch.Export.image.toDrive(
            image=raster, description=prefix+"_evidence", folder=cfg.drive_folder,
            fileNamePrefix=prefix+"_evidence", region=product.region,
            crs=cfg.crs, scale=cfg.scale, maxPixels=cfg.max_pixels,
            fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True, "noData": NODATA}),
        "candidate_polygons_geojson": ee.batch.Export.table.toDrive(
            collection=product.targets, description=prefix+"_polygons",
            folder=cfg.drive_folder, fileNamePrefix=prefix+"_polygons", fileFormat="GeoJSON")
    }
    strict = product.targets.filter(ee.Filter.eq("evidence_tier", 3))
    count = product.targets.size().getInfo()
    strict_count = strict.size().getInfo()
    product.report["target_polygon_count"] = count
    product.report["strict_evidence_polygon_count"] = strict_count
    product.report["strict_evidence_label"] = "MULTI_EVIDENCE_CANDIDATE_NOT_FIELD_VALIDATED"
    # An empty strict KML is not advertised as a successful target product.
    if strict_count:
        jobs["strict_polygons_kml"] = ee.batch.Export.table.toDrive(
            collection=strict, description=prefix+"_strict", folder=cfg.drive_folder,
            fileNamePrefix=prefix+"_strict", fileFormat="KML")
    if count:
        def navpoint(feature):
            f = ee.Feature(feature)
            point = ee.Geometry.Point([f.get("nav_longitude"), f.get("nav_latitude")])
            return ee.Feature(point, f.toDictionary())
        nav = product.targets.map(navpoint)
        jobs["navigation_kml"] = ee.batch.Export.table.toDrive(
            collection=nav, description=prefix+"_navigation", folder=cfg.drive_folder,
            fileNamePrefix=prefix+"_navigation", fileFormat="KML")
        jobs["navigation_csv"] = ee.batch.Export.table.toDrive(
            collection=nav, description=prefix+"_navigation_csv", folder=cfg.drive_folder,
            fileNamePrefix=prefix+"_navigation", fileFormat="CSV",
            selectors=["target_id", "nav_latitude", "nav_longitude", "area_m2", "evidence_tier",
                       "mean_persistence", "mean_valid_dates", "classification",
                       "field_validated", "analysis_pixel_m", "dem_native_m"])
    return jobs


def submit_exports(jobs, report):
    """No blind retries: save each returned task id before resubmission decisions."""
    report["exports"] = {}
    for name, task in jobs.items():
        try:
            task.start()
            report["exports"][name] = task.status()
        except Exception as exc:
            # Keep prior ids for partial submission recovery. Never mark a job
            # completed merely because start() returned successfully.
            report["exports"][name] = {"state": "SUBMISSION_FAILED", "error": str(exc)}
            raise
    return report["exports"]

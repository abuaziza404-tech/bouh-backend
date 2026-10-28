"""Bounded orchestration: discovery -> QA -> PCA -> local evidence -> exports.

No automatic downgrade on asset permission errors or failed server operations.
Only scientifically optional absent inputs have explicit degraded behavior.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import ee
from .config import Config
from .numeric import QualityGateError
from .masking import (BANDS, terrain_context, basic_mask_scene, calibrate_environment,
                      mask_environment, safe_divide)
from .statistics import fit_feature_pca, apply_pca, local_anomaly, reduction
from .structures import structure_layers, geology_layers

LOG = logging.getLogger("bouh")


@dataclass
class Product:
    evidence: object
    targets: object
    region: object
    report: dict


def covered_fraction(mask, region, cfg):
    """Denominator includes all AOI pixels; zero is explicit QA only, not SR."""
    value = reduction(mask.unmask(0, sameFootprint=False).rename("covered"), ee.Reducer.mean(),
                      region, cfg).getInfo().get("covered")
    return float(value or 0)


def discover_scenes(region, cfg, report):
    sr = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region).filterDate(cfg.start, cfg.end)
          .filter(ee.Filter.notNull(["MEAN_SOLAR_AZIMUTH_ANGLE", "MEAN_SOLAR_ZENITH_ANGLE"]))
          .sort("system:time_start"))
    count = sr.size().getInfo()
    if count == 0:
        raise QualityGateError("No Sentinel-2 L2A scenes for this AOI/date range")
    if count > cfg.max_scenes:
        raise QualityGateError(f"{count} granules exceed max_scenes={cfg.max_scenes}; narrow dates/tile")
    cloud = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    ids = sr.aggregate_array("system:index").getInfo()
    pairs = []
    for scene_id in ids:
        image = ee.Image("COPERNICUS/S2_SR_HARMONIZED/"+scene_id)
        entry = {"asset_id": "COPERNICUS/S2_SR_HARMONIZED/"+scene_id,
                 "access_status": "METADATA_INSPECTED",
                 "raw_download_status": "NOT_DOWNLOADED",
                 "raw_sha256": None,
                 "raw_sha256_reason": "Earth Engine managed input; source raster bytes not downloaded"}
        prob = cloud.filter(ee.Filter.eq("system:index", scene_id))
        if prob.size().getInfo() == 0:
            entry.update(status="SKIPPED", reason="MISSING_CLOUD_PROBABILITY")
            report["scenes"].append(entry)
            continue
        metadata = image.toDictionary([
            "PRODUCT_ID", "system:time_start", "MGRS_TILE", "PROCESSING_BASELINE",
            "MEAN_SOLAR_AZIMUTH_ANGLE", "MEAN_SOLAR_ZENITH_ANGLE"
        ]).getInfo()
        az, z = metadata.get("MEAN_SOLAR_AZIMUTH_ANGLE"), metadata.get("MEAN_SOLAR_ZENITH_ANGLE")
        if az is None or z is None or not (0 <= az <= 360 and 0 < z < 90):
            entry.update(status="SKIPPED", reason="INVALID_SOLAR_GEOMETRY")
            report["scenes"].append(entry)
            continue
        entry.update(metadata=metadata, cloud_asset="COPERNICUS/S2_CLOUD_PROBABILITY/"+scene_id)
        report["scenes"].append(entry)
        pairs.append((image, ee.Image(prob.first()), entry))
    if not pairs:
        raise QualityGateError("All scenes lack required QA or solar geometry")
    return pairs


def process(cfg: Config, report=None) -> Product:
    report = report if report is not None else {}
    cfg.validate()
    region = ee.Geometry(cfg.aoi_geojson)
    area = region.area(1).getInfo()/1e6
    if not 0 < area <= cfg.max_aoi_km2:
        raise QualityGateError("AOI too large/invalid: split into local processing units")
    # Enforce the declared UTM zone and hemisphere rather than accepting a CRS
    # whose meter-based neighborhoods would be materially wrong for the AOI.
    lon, lat = region.centroid(1).coordinates().getInfo()
    zone = min(60, int((lon+180)//6)+1)
    expected = f"EPSG:{32600+zone if lat >= 0 else 32700+zone}"
    if cfg.crs != expected:
        raise QualityGateError(f"AOI centroid requires {expected}, got {cfg.crs}")
    report.update({"config": cfg.record(), "created_utc": datetime.now(timezone.utc).isoformat(),
              "scenes": [], "warnings": [], "models": {}, "area_km2": area,
              "state": "PROCESSING_STARTED", "field_validation": "NOT_PERFORMED",
              "high_confidence_gold_claim": False,
              "evidence_independence": "Ferric/hydroxyl PCs share Sentinel-2; NOT independent sensors"})
    terrain = terrain_context(region, cfg)
    report["dem"] = {"collection": cfg.dem_collection, "type": "DSM", "native_m": 30,
                     "vertical_reference": "EGM2008", "shadow_crs": "EPSG:3857",
                     "shadow_padding_m": terrain["padding_m"],
                     "solar_geometry": "PER_GRANULE_MEAN_ANGLES"}
    # Process a halo for neighborhood statistics before clipping the output AOI.
    work = region.buffer(cfg.local_radius_m + cfg.structure_buffer_m + 100)
    # terrain_context has a substantially larger cast-shadow halo.
    if cfg.local_radius_m + cfg.structure_buffer_m + 100 >= terrain["padding_m"]:
        raise QualityGateError("Shadow halo must exceed analysis halo")
    masked = []
    for image, cloud, entry in discover_scenes(work, cfg, report):
        clean = basic_mask_scene(image, cloud, terrain, cfg).clip(work)
        coverage = covered_fraction(clean.select("B11").mask(), region, cfg)
        entry["basic_coverage_aoi"] = coverage
        if coverage < cfg.min_scene_coverage:
            entry.update(status="SKIPPED", reason="LOW_AOI_VALID_COVERAGE")
            continue
        entry["status"] = "BASIC_MASK_EXECUTED"
        masked.append(clean)
    if not masked:
        raise QualityGateError("No scene passes AOI quality coverage")
    basic_collection = ee.ImageCollection.fromImages(masked)
    thresholds = calibrate_environment(
        basic_collection.median().setDefaultProjection(crs=cfg.crs, scale=cfg.scale), work, cfg)
    report["environment_thresholds"] = thresholds
    processed = basic_collection.map(lambda im: mask_environment(ee.Image(im), thresholds, cfg))
    # Mosaic overlapping granules on each UTC date. One date = one observation;
    # overlapping MGRS granules cannot inflate persistence.
    dates = ee.List(processed.aggregate_array("bouh_date")).distinct().sort().getInfo()
    daily = []
    for day in dates:
        day_image = (processed.filter(ee.Filter.eq("bouh_date", day)).sort("system:index")
                     .mosaic().setDefaultProjection(crs=cfg.crs, scale=cfg.scale).set("bouh_date", day))
        f = covered_fraction(day_image.select("B11").mask(), region, cfg)
        report.setdefault("dates", []).append({"utc_date": day, "valid_fraction": f})
        if f >= cfg.min_scene_coverage:
            daily.append(day_image)
    if len(daily) < cfg.min_dates:
        raise QualityGateError(f"Only {len(daily)} valid acquisition dates; need {cfg.min_dates}")
    observations = ee.ImageCollection.fromImages(daily).select(BANDS).count().reduce(ee.Reducer.min())
    composite = (ee.ImageCollection.fromImages(daily).select(BANDS).median()
                 .setDefaultProjection(crs=cfg.crs, scale=cfg.scale))
    composite = composite.updateMask(observations.gte(cfg.min_observations))
    coverage = covered_fraction(composite.select("B11").mask(), region, cfg)
    report["eligible_coverage_fraction"] = coverage
    if coverage < cfg.min_eligible_coverage:
        raise QualityGateError("Insufficient multi-date AOI coverage")
    models = {}
    for kind in ("ferric", "hydroxyl"):
        models[kind] = fit_feature_pca(composite, work, kind, cfg)
    report["models"] = models
    # Identical transforms on every date; thresholds remain spatially local.
    per_date = []
    for image in daily:
        layers = []
        for kind in models:
            pc = apply_pca(image, models[kind], cfg)
            layers.append(local_anomaly(pc, kind, cfg))
        both = ee.Image.cat(layers)
        f, h = both.select("ferric_anomaly"), both.select("hydroxyl_anomaly")
        joint = f.And(h).rename("joint")
        # This validity band follows all numerical/local-statistics masks.
        valid = joint.mask().rename("date_valid")
        both = both.addBands(joint).addBands(valid)
        per_date.append(both)
    temporal = ee.ImageCollection.fromImages(per_date)
    valid_count = temporal.select("date_valid").sum().rename("valid_date_count")
    joint_count = temporal.select("joint").sum().rename("joint_date_count")
    persistence = safe_divide(joint_count, valid_count, cfg.denominator_epsilon).rename("persistence")
    marginal_layers = []
    for kind in ("ferric", "hydroxyl"):
        count = temporal.select(kind+"_anomaly").count().rename(kind+"_valid_date_count")
        hits = temporal.select(kind+"_anomaly").sum()
        fraction = safe_divide(hits, count, cfg.denominator_epsilon).rename(kind+"_persistence")
        marginal_layers.extend([count, fraction])
    persistent = (persistence.gte(cfg.min_persistence)
                  .And(valid_count.gte(cfg.min_observations)))
    structures = structure_layers(terrain, work, cfg)
    geology = geology_layers(work, cfg)
    redbed = geology.select("redbed_excluded")
    spectral = persistent.And(redbed.Not())
    # A known exclusion suppresses targets. Unknown lithology is flagged, never
    # silently classified as non-redbed or promoted to the strict evidence tier.
    weak = spectral.And(structures.select("dem_candidate_near"))
    strict = (spectral.And(structures.select("reviewed_structure_near"))
              .And(geology.select("geology_known")))
    # 0=no selected anomaly, 1=spectral, 2=DEM-edge-supported,
    # 3=reviewed-structure + mapped lithology supported. NO gold probabilities.
    tier = (ee.Image.constant(0).setDefaultProjection(crs=cfg.crs, scale=cfg.scale).where(spectral, 1).where(weak, 2).where(strict, 3)
            .updateMask(valid_count.gte(cfg.min_observations)).rename("evidence_tier"))
    # Remove isolated components from target products, retain diagnostics intact.
    candidate = tier.gt(0)
    coherent = candidate.selfMask().connectedPixelCount(256, True).gte(cfg.min_component_pixels)
    tier = tier.where(candidate.And(coherent.unmask(0).Not()), 0)
    diagnostic = ee.Image.cat(marginal_layers + [
        tier, persistence, valid_count, joint_count, observations.rename("sr_observation_count"),
        structures, geology,
        temporal.select("ferric_robust_z").median().rename("ferric_robust_z_median"),
        temporal.select("hydroxyl_robust_z").median().rename("hydroxyl_robust_z_median"),
        ee.ImageCollection.fromImages(daily).select("moisture_flag").max().rename("moisture_flag_any")
    ]).clip(region)
    targets = vectorize_targets(tier.clip(region), diagnostic, region, cfg)
    if not cfg.reviewed_structures_asset:
        report["warnings"].append("NO_REVIEWED_STRUCTURES: tier 3 disabled; DEM edges are not faults")
    if not (cfg.redbeds_asset and cfg.geology_coverage_asset):
        report["warnings"].append("UNKNOWN_LITHOLOGY: no validated red-bed discrimination; tier 3 disabled")
    report["warnings"].extend([
        "No specific mineral, gold grade, burial depth, nugget size or detector response inferred",
        "Placer hydrology/source-to-sink modelling NOT performed; these are bedrock-alteration candidates",
        "Local quantile/MAD and PC loading gates are engineering defaults pending field calibration",
        "Raw scenes not downloaded or byte-hashed; exported products require separate hashes after download"
    ])
    report["state"] = "GEE_GRAPH_CONSTRUCTED_QA_AND_PCA_SUMMARIES_EVALUATED"
    report["config_sha256"] = hashlib.sha256(json.dumps(cfg.record(), sort_keys=True).encode()).hexdigest()
    return Product(diagnostic, targets, region, report)


def vectorize_targets(tier, evidence, region, cfg):
    """Export full target polygons and an internal navigation point per polygon.

    The point is nearest eligible pixel to the polygon centroid, not a subpixel
    gold location. Preserve polygon extent and georeferencing uncertainty.
    """
    labels = tier.toInt().selfMask().rename("tier")
    polygons = labels.reduceToVectors(
        reducer=ee.Reducer.countEvery(), geometry=region, scale=cfg.scale,
        crs=cfg.crs, geometryType="polygon", eightConnected=True,
        labelProperty="evidence_tier", maxPixels=cfg.max_pixels, tileScale=cfg.tile_scale)

    def describe(feature):
        f = ee.Feature(feature); geom = f.geometry()
        center = geom.centroid(1)
        eligible = labels.eq(ee.Number(f.get("evidence_tier")))
        # A small local geographic-distance approximation chooses a pixel;
        # it is not used for area, buffering, or reported location accuracy.
        lonlat = ee.Image.pixelLonLat()
        c = center.coordinates()
        cos_lat = ee.Number(c.get(1)).multiply(math.pi/180).cos()
        d2 = (lonlat.select("longitude").subtract(ee.Number(c.get(0))).multiply(cos_lat).pow(2)
              .add(lonlat.select("latitude").subtract(ee.Number(c.get(1))).pow(2)))
        d2 = d2.updateMask(eligible).rename("distance2")
        minimum = reduction(d2, ee.Reducer.min(), geom, cfg).get("distance2")
        points = (lonlat.updateMask(d2.lte(ee.Number(minimum).add(1e-14)))
                  .sample(region=geom, scale=cfg.scale, projection=cfg.crs,
                          geometries=True, tileScale=cfg.tile_scale, dropNulls=True)
                  .sort("longitude").limit(1))
        point = ee.Feature(points.first())
        p = point.geometry().coordinates()
        stats = reduction(evidence.select(["persistence", "valid_date_count"]),
                          ee.Reducer.mean(), geom, cfg)
        return f.set({"target_id": ee.String("BOUH_").cat(ee.Number(p.get(1)).format("%.6f")).cat("_").cat(ee.Number(p.get(0)).format("%.6f")), "area_m2": geom.area(1),
                      "nav_longitude": p.get(0), "nav_latitude": p.get(1),
                      "analysis_pixel_m": cfg.scale, "dem_native_m": 30,
                      "classification": "UNVALIDATED_FIELD_RECONNAISSANCE",
                      "gold_probability": "NOT_ESTIMATED",
                      "field_validated": False,
                      "mean_persistence": stats.get("persistence"),
                      "mean_valid_dates": stats.get("valid_date_count")})
    return polygons.map(describe)

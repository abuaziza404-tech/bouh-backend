"""Per-granule DEM shadow and environmental masking on explicit grids."""
import math
import ee
from .numeric import QualityGateError

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def safe_divide(num, den, epsilon=1e-6):
    """An EE division with a finite-value contract and missing-data mask.

    Test both inputs, substitute an innocuous denominator BEFORE division,
    then mask invalid output. Missingness never becomes a zero observation.
    abs<1e30 rejects infinities; x==x rejects NaN. Epsilon is numerical only.
    """
    num, den = ee.Image(num).toDouble(), ee.Image(den).toDouble()
    valid = (num.eq(num).And(den.eq(den)).And(num.abs().lt(1e30))
             .And(den.abs().lt(1e30)).And(den.abs().gt(epsilon)))
    result = num.divide(den.where(valid.Not(), 1))
    return result.updateMask(valid.And(result.eq(result)).And(result.abs().lt(1e30)))


def terrain_context(aoi, cfg):
    """DSM, not DTM; retain surrounding topography for cast-shadow rays."""
    padding = cfg.shadow_neighborhood_px * cfg.dem_scale * math.sqrt(2) + 1000
    region = aoi.buffer(padding)
    dem_col = ee.ImageCollection(cfg.dem_collection).filterBounds(region)
    if dem_col.size().getInfo() == 0:
        raise QualityGateError("No DEM tiles: terrain-shadow processing is blocked")
    dem = dem_col.select("DEM").mosaic().resample("bilinear")
    merc = dem.reproject(crs="EPSG:3857", scale=cfg.dem_scale).clip(region)
    local = dem.reproject(crs=cfg.crs, scale=cfg.dem_scale).clip(region)
    terrain = ee.Terrain.products(local)
    return {"dem_mercator": merc, "dem_local": local,
            "slope": terrain.select("slope"), "aspect": terrain.select("aspect"),
            "padding_m": padding}


def dem_illumination(image, terrain, cfg):
    """Cast shadow AND local slope self-shadow, using scene-mean solar angles.

    Small processing tiles limit scene-mean angle approximation. No reflectance
    is used to infer terrain shadows; dark mafic rocks remain eligible.
    No topographic radiometric correction is claimed by this mask.
    """
    az = ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
    z = ee.Number(image.get("MEAN_SOLAR_ZENITH_ANGLE"))
    rad = math.pi/180
    slope = terrain["slope"].multiply(rad)
    aspect = terrain["aspect"].multiply(rad)
    cosine = (slope.cos().multiply(z.multiply(rad).cos())
              .add(slope.sin().multiply(z.multiply(rad).sin())
                   .multiply(ee.Image.constant(az.multiply(rad)).subtract(aspect).cos())))
    # GEE specifies Mercator, NOT geographic or UTM, for this ray algorithm.
    cast_lit = ee.Terrain.hillShadow(
        terrain["dem_mercator"], az, z, cfg.shadow_neighborhood_px, False
    ).eq(1).reproject(crs=cfg.crs, scale=cfg.scale)
    return cast_lit.And(cosine.gt(cfg.min_illumination_cosine)).rename("illuminated")


def basic_mask_scene(sr, cloud, terrain, cfg):
    """Cloud probability + SCL + DEM. QA remains nearest-neighbor categorical."""
    scl = sr.select("SCL")
    allowed = scl.eq(5)
    if cfg.allow_unclassified_scl:
        allowed = allowed.Or(scl.eq(7))
    # Dilate cloud/shadow/defect classes to reduce adjacency contamination.
    bad = (scl.eq(0).Or(scl.eq(1)).Or(scl.eq(3)).Or(scl.gte(8))
           .Or(cloud.select("probability").gte(cfg.cloud_probability_max)))
    bad = bad.focalMax(radius=cfg.cloud_buffer_m, units="meters")
    good = allowed.And(bad.Not()).And(dem_illumination(sr, terrain, cfg))
    good = good.And(sr.select("B8A").mask()).And(sr.select("B9").mask())
    # Input is SR_HARMONIZED: only 1e-4 scaling, NO second baseline offset.
    optical = sr.select(BANDS).multiply(0.0001).toDouble().updateMask(good)
    # Explicit averaging of native 10m reflectance to analysis grid; SWIR stays
    # at its native 20m support. This never creates 10m SWIR information.
    out = []
    for band in BANDS:
        b = optical.select(band)
        if band in ("B02", "B03", "B04", "B08"):
            b = b.reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
        b = b.reproject(crs=cfg.crs, scale=cfg.scale)
        out.append(b.rename(band))
    out = ee.Image.cat(out)
    valid = out.mask().reduce(ee.Reducer.min())
    valid = valid.And(out.gte(0).reduce(ee.Reducer.min()))
    return (out.updateMask(valid).copyProperties(sr, sr.propertyNames())
            .set("bouh_date", ee.Date(sr.get("system:time_start")).format("YYYY-MM-dd")))


def environment_indices(image, cfg):
    nir, red = image.select("B08"), image.select("B04")
    green, sw1 = image.select("B03"), image.select("B11")
    eps = cfg.denominator_epsilon
    return ee.Image.cat([
        safe_divide(nir.subtract(red), nir.add(red), eps).rename("ndvi"),
        safe_divide(green.subtract(nir), green.add(nir), eps).rename("ndwi"),
        safe_divide(green.subtract(sw1), green.add(sw1), eps).rename("mndwi"),
        safe_divide(nir.subtract(sw1), nir.add(sw1), eps).rename("ndmi")
    ])


def calibrate_environment(composite, region, cfg):
    """Local quantiles bounded by disclosed expert ranges, NOT ground calibration.

    Moisture is flagged by default; hydrothermal materials can share moisture
    index responses. Never call an NDWI threshold a wet-clay mineral detector.
    """
    bands = ["ndvi", "ndwi", "ndmi"]
    indices = environment_indices(composite, cfg)
    stats = indices.select(bands).reduceRegion(
        reducer=ee.Reducer.percentile([90]), geometry=region,
        scale=cfg.scale, crs=cfg.crs, maxPixels=cfg.max_pixels,
        tileScale=cfg.tile_scale).getInfo()
    result = {}
    for band, limits in zip(bands, [cfg.env_ndvi_bounds, cfg.env_ndwi_bounds,
                                   cfg.env_moisture_bounds]):
        value = stats.get(band+"_p90")
        if value is None or not math.isfinite(value):
            raise QualityGateError("Insufficient data for environmental calibration")
        result[band] = min(max(value, limits[0]), limits[1]) if cfg.adaptive_environment else limits[0]
    result["policy"] = "LOCAL_P90_BOUNDED_UNVALIDATED_DEFAULTS"
    return result


def mask_environment(image, thresholds, cfg):
    idx = environment_indices(image, cfg)
    good = idx.select("ndvi").lte(thresholds["ndvi"])
    water = idx.select("ndwi").gt(thresholds["ndwi"]).And(idx.select("mndwi").gt(0))
    moist = idx.select("ndmi").gt(thresholds["ndmi"]).rename("moisture_flag")
    good = good.And(water.Not())
    if cfg.reject_moisture_flags:
        good = good.And(moist.Not())
    return image.addBands(idx).addBands(moist).updateMask(good)

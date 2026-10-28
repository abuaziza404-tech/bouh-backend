"""DEM structural candidates are explicitly separate from reviewed faults."""
import ee
from .statistics import reduction
from .numeric import QualityGateError


def structure_layers(terrain, region, cfg):
    dem = terrain["dem_local"]
    empty = ee.Image.constant(0).clip(region).reproject(crs=cfg.crs, scale=cfg.dem_scale)
    candidates = empty.rename("dem_edge_candidate")
    if cfg.derive_dem_edges:
        # Multiple virtual lighting directions reduce single-aspect bias.
        # Sobel directional gradients + consensus select elongated terrain edges;
        # these are NOT fault traces and can include channels/contacts/roads.
        votes = []
        for az in (0, 45, 90, 135):
            shade = ee.Terrain.hillshade(dem, az, 45).toDouble()
            gx = shade.convolve(ee.Kernel.sobel())
            gy = shade.convolve(ee.Kernel.sobel().rotate(1))
            strength = gx.pow(2).add(gy.pow(2)).sqrt().rename("edge_strength")
            threshold = reduction(strength, ee.Reducer.percentile([cfg.dem_edge_quantile]),
                                  region, cfg, cfg.dem_scale).getInfo()
            value = next(iter(threshold.values()), None)
            if value is None:
                raise QualityGateError("DEM directional edge statistics unavailable")
            votes.append(strength.gt(value))
        candidates = (ee.ImageCollection.fromImages(votes).sum()
                      .setDefaultProjection(crs=cfg.crs, scale=cfg.dem_scale).gte(2)
                      .rename("dem_edge_candidate"))
        # Filter isolated edge noise; not a geological validation gate.
        candidates = candidates.And(candidates.selfMask().connectedPixelCount(32, True).gte(4))
    candidate_near = candidates.unmask(0).focalMax(
        radius=cfg.structure_buffer_m, units="meters").rename("dem_candidate_near")
    reviewed = empty.rename("reviewed_structure_near")
    if cfg.reviewed_structures_asset:
        fc = ee.FeatureCollection(cfg.reviewed_structures_asset).filterBounds(region)
        if fc.size().getInfo() == 0:
            raise QualityGateError("Reviewed-structure asset has no features in processing region")
        line_raster = empty.paint(fc, 1, 1)
        reviewed = line_raster.focalMax(radius=cfg.structure_buffer_m,
                                        units="meters").rename("reviewed_structure_near")
    return ee.Image.cat([candidates.unmask(0), candidate_near.unmask(0), reviewed.unmask(0)])


def geology_layers(region, cfg):
    """Unknown coverage never becomes 'not red bed'. Require a coverage raster."""
    zero = ee.Image.constant(0).clip(region)
    if not cfg.geology_coverage_asset:
        return ee.Image.cat([zero.rename("geology_known"), zero.rename("redbed_excluded")])
    coverage = ee.Image(cfg.geology_coverage_asset).select(0)
    known = coverage.eq(1).unmask(0).rename("geology_known")
    if cfg.redbeds_asset:
        raw = ee.Image(cfg.redbeds_asset).select(0)
        # Within declared coverage, a masked exclusion pixel is unknown too.
        known = known.And(raw.mask()).rename("geology_known")
        excluded = raw.eq(1).unmask(0).rename("redbed_excluded")
    else:
        # Cannot assert red-bed absence without the explicit exclusion layer.
        known = zero.rename("geology_known")
        excluded = zero.rename("redbed_excluded")
    return ee.Image.cat([known.unmask(0), excluded])

"""PCA fitting, signature selection, robust spatial-local anomaly gates."""
import math
import numpy as np
import ee
from .masking import safe_divide
from .numeric import QualityGateError, select_component

# Reflectance prototypes, NOT calibrated mineral endmembers.
SIGNATURES = {
    "ferric": {"bands": ["B02", "B04", "B08", "B11"],
               "positive": "B04", "negative": "B02"},
    "hydroxyl": {"bands": ["B02", "B08", "B11", "B12"],
                 "positive": "B11", "negative": "B12"}
}


def reduction(image, reducer, region, cfg, scale=None):
    return image.reduceRegion(reducer=reducer, geometry=region,
                              scale=scale or cfg.scale, crs=cfg.crs,
                              maxPixels=cfg.max_pixels, tileScale=cfg.tile_scale)


def fit_feature_pca(composite, region, kind, cfg):
    """Correlation PCA (z-standardized), explicit means/std and degeneracy gates.

    Only small summaries come to Python. Covariance and image projection remain
    in GEE. Model is fit once per spatial/statistical unit and applied unchanged
    across dates, so persistence is not based on independently rotated PCs.
    """
    spec = SIGNATURES[kind]; bands = spec["bands"]
    x = composite.select(bands)
    x = x.updateMask(x.mask().reduce(ee.Reducer.min()))
    reducer = (ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)
               .combine(ee.Reducer.count(), sharedInputs=True))
    stats = reduction(x, reducer, region, cfg, cfg.covariance_scale).getInfo()
    means, stds = [], []
    for b in bands:
        mean, std, count = stats.get(b+"_mean"), stats.get(b+"_stdDev"), stats.get(b+"_count", 0)
        if (mean is None or std is None or count < cfg.min_pca_pixels or
            not math.isfinite(mean) or not math.isfinite(std) or std <= cfg.min_std):
            raise QualityGateError(f"{kind}: insufficient/constant band {b}")
        means.append(mean); stds.append(std)
    mu = ee.Image.constant(means).rename(bands)
    sd = ee.Image.constant(stds).rename(bands)
    z = safe_divide(x.subtract(mu), sd, cfg.denominator_epsilon)
    cov = reduction(z.toArray(), ee.Reducer.centeredCovariance(), region,
                    cfg, cfg.covariance_scale).getInfo().get("array")
    if cov is None:
        raise QualityGateError(f"{kind}: covariance unavailable")
    # Explicit symmetric covariance check prior to GEE eigendecomposition.
    c = np.asarray(cov, float)
    if not np.isfinite(c).all() or not np.allclose(c, c.T, atol=1e-7):
        raise QualityGateError(f"{kind}: non-finite/asymmetric covariance")
    eig = np.asarray(ee.Array(cov).eigen().getInfo(), float)
    choice = select_component(eig[:, 0], eig[:, 1:], bands,
                              spec["positive"], spec["negative"],
                              cfg.min_pc_cosine, cfg.min_pc_pair_energy,
                              min_gap=cfg.min_pc_gap)
    return {"kind": kind, "bands": bands, "means": means, "stds": stds,
            "covariance": cov, "eigenvalues": eig[:, 0].tolist(),
            "eigenvectors_rows": eig[:, 1:].tolist(), "choice": choice.record(),
            "method": "CORRELATION_PCA_FEATURE_ORIENTED",
            "note": "FERRIC/HYDROXYL PROXIES; NOT SPECIFIC MINERAL IDENTIFICATION"}


def apply_pca(image, model, cfg):
    bands = model["bands"]
    x = image.select(bands)
    z = safe_divide(x.subtract(ee.Image.constant(model["means"]).rename(bands)),
                    ee.Image.constant(model["stds"]).rename(bands),
                    cfg.denominator_epsilon)
    row = model["choice"]["loadings"]  # sign already oriented
    pc = z.multiply(ee.Image.constant(row).rename(bands)).reduce(ee.Reducer.sum())
    pc = pc.updateMask(z.mask().reduce(ee.Reducer.min()))
    return pc.rename(model["kind"])


def local_anomaly(image, name, cfg):
    """Local median + scaled MAD AND upper quantile, with a sample-count gate.

    MAD is the local median absolute deviation around each window's median:
    derive via masked neighborhood arrays, NOT by filtering per-pixel residuals.
    Neighborhood size is explicit; keep processing units small.
    Zero-MAD windows fail closed. Thresholds are heuristics pending validation.
    """
    x = image.select(name)
    # A square local window allows exact masked-array MAD without creating
    # thousands of named bands. Missing neighbors are removed, not set to zero.
    radius = math.ceil(cfg.local_radius_m/cfg.scale)
    width = 2*radius+1
    kernel = ee.Kernel.square(radius, "pixels", normalize=False)
    missing = -1e30
    neighbors = x.neighborhoodToArray(kernel, defaultValue=missing)
    neighbors = neighbors.arrayReshape(ee.Image.constant(ee.Array([width*width])), 1)
    neighbors = neighbors.arrayMask(neighbors.neq(missing))
    med = neighbors.arrayReduce(ee.Reducer.median(), [0]).arrayGet([0])
    mad = neighbors.subtract(med).abs().arrayReduce(ee.Reducer.median(), [0]).arrayGet([0])
    q = neighbors.arrayReduce(ee.Reducer.percentile([cfg.anomaly_quantile]), [0]).arrayGet([0])
    count = neighbors.arrayLength(0)
    robust_z = safe_divide(x.subtract(med), mad.multiply(1.4826), cfg.denominator_epsilon)
    valid = count.gte(cfg.min_local_pixels).And(mad.gt(cfg.denominator_epsilon))
    anomaly = x.gt(q).And(robust_z.gte(cfg.anomaly_mad_z)).updateMask(valid)
    return ee.Image.cat([anomaly.rename(name+"_anomaly"),
                         robust_z.rename(name+"_robust_z"),
                         q.rename(name+"_local_quantile"),
                         count.rename(name+"_local_count")])

"""Versioned, explicit processing configuration. No credentials in this file."""
from dataclasses import dataclass, field, asdict
from datetime import date
import math
from .numeric import QualityGateError


@dataclass
class Config:
    project: str
    aoi_geojson: dict
    start: str
    end: str                      # exclusive, per Earth Engine filterDate
    crs: str = "EPSG:32637"
    scale: int = 20
    dem_scale: int = 30
    dem_collection: str = "COPERNICUS/DEM/GLO30_2024_1"
    max_aoi_km2: float = 400.0     # split regional jobs into explicit local units
    max_scenes: int = 24
    min_dates: int = 3
    min_observations: int = 3
    min_persistence: float = 0.67
    min_scene_coverage: float = 0.20
    min_eligible_coverage: float = 0.30
    cloud_probability_max: int = 35
    cloud_buffer_m: int = 60
    shadow_neighborhood_px: int = 240
    min_illumination_cosine: float = 0.10
    adaptive_environment: bool = True
    env_ndvi_bounds: tuple = (0.15, 0.30)
    env_ndwi_bounds: tuple = (0.00, 0.15)
    env_moisture_bounds: tuple = (0.10, 0.35)
    reject_moisture_flags: bool = False
    allow_unclassified_scl: bool = False
    covariance_scale: int = 20
    min_pca_pixels: int = 500
    min_std: float = 1e-5
    denominator_epsilon: float = 1e-6
    anomaly_quantile: float = 98.0
    anomaly_mad_z: float = 2.5
    local_radius_m: int = 400
    min_local_pixels: int = 100
    min_component_pixels: int = 4
    min_pc_cosine: float = 0.55
    min_pc_pair_energy: float = 0.55
    min_pc_gap: float = 0.05
    reviewed_structures_asset: str | None = None
    # Binary rasters: 1=known excluded red beds; 1=reviewed mapped coverage.
    redbeds_asset: str | None = None
    geology_coverage_asset: str | None = None
    geology_reference: str | None = None
    structure_buffer_m: int = 120
    derive_dem_edges: bool = True
    dem_edge_quantile: float = 95.0
    tile_scale: int = 4
    max_pixels: int = 10_000_000
    drive_folder: str = "BOUH_Terrain"
    start_exports: bool = False
    # Area statistics are deliberately spatially local, not global percentiles.
    schema_version: str = "bouh-evidence-0.1.0"

    def validate(self):
        if not self.project.strip():
            raise ValueError("An Earth Engine enabled Cloud project is required")
        if date.fromisoformat(self.end) <= date.fromisoformat(self.start):
            raise ValueError("end must be after start")
        if self.scale < 20 or self.dem_scale < 30:
            raise ValueError("Do not imply SWIR<20m or DSM<30m native resolution")
        if not self.crs.startswith(("EPSG:326", "EPSG:327")):
            raise ValueError("Use the appropriate metric UTM CRS for the local AOI")
        if self.aoi_geojson.get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("AOI must be a GeoJSON Polygon or MultiPolygon geometry")
        if self.min_dates < 2 or self.min_observations < 2:
            raise ValueError("At least two independent acquisition dates required")
        if self.max_scenes < self.min_dates:
            raise ValueError("max_scenes < min_dates")
        if not 0 < self.min_persistence <= 1:
            raise ValueError("Invalid persistence fraction")
        if not 50 < self.anomaly_quantile < 100:
            raise ValueError("Invalid anomaly quantile")
        for name in ("min_scene_coverage", "min_eligible_coverage"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"Invalid {name}")
        if not 0 < self.cloud_probability_max <= 100:
            raise ValueError("Invalid cloud threshold")
        for name in ("env_ndvi_bounds", "env_ndwi_bounds", "env_moisture_bounds"):
            lo, hi = getattr(self, name)
            if not (-1 <= lo <= hi <= 1):
                raise ValueError(f"Invalid {name}")
        for name in ("scale", "dem_scale", "max_aoi_km2", "shadow_neighborhood_px",
                     "denominator_epsilon", "min_std", "local_radius_m",
                     "min_local_pixels", "min_pca_pixels", "max_pixels",
                     "anomaly_mad_z", "structure_buffer_m", "tile_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")
        if self.redbeds_asset and not self.geology_coverage_asset:
            raise ValueError("Red-bed raster requires explicit geology coverage")
        if self.geology_coverage_asset and not self.geology_reference:
            raise ValueError("Geology provenance/reference is required")
        return self

    def record(self):
        return asdict(self)

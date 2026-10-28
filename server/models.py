"""Bounded request contract: a small bbox or axis-aligned GeoJSON Polygon."""
from datetime import date
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    bbox: tuple[float, float, float, float] | None = None  # W,S,E,N
    polygon: dict | None = None
    start_date: date
    end_date: date
    min_tier: Literal[1, 2, 3] = 1
    include_proxy_layers: bool = True

    @model_validator(mode="after")
    def validate_region(self):
        if (self.bbox is None) == (self.polygon is None):
            raise ValueError("Provide exactly one of bbox or polygon")
        if self.polygon is not None:
            if self.polygon.get("type") != "Polygon":
                raise ValueError("Only a rectangular GeoJSON Polygon is supported")
            rings = self.polygon.get("coordinates", [])
            if len(rings) != 1 or len(rings[0]) != 5:
                raise ValueError("Expected one closed rectangular ring")
            ring = rings[0]
            if any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in ring):
                raise ValueError("Expected [longitude, latitude] pairs")
            if any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v)
                   for p in ring for v in p):
                raise ValueError("Invalid coordinate")
            xs, ys = sorted(set(p[0] for p in ring)), sorted(set(p[1] for p in ring))
            if len(xs) != 2 or len(ys) != 2 or ring[0] != ring[-1]:
                raise ValueError("Polygon must be a closed axis-aligned rectangle")
            corners = {(x,y) for x in xs for y in ys}
            if {tuple(p) for p in ring[:-1]} != corners:
                raise ValueError("Invalid rectangular corners")
            if any(a[0] != b[0] and a[1] != b[1] for a,b in zip(ring,ring[1:])):
                raise ValueError("Crossed polygon edges are invalid")
            self.bbox = (xs[0],ys[0],xs[1],ys[1])
        w,s,e,n = self.bbox
        if not (-180 <= w < e <= 180 and -60 <= s < n <= 75):
            raise ValueError("Invalid bbox or unsupported polar/antimeridian extent")
        if e-w > .2 or n-s > .2:
            raise ValueError("Map extent too large: each bbox side must be <=0.2 degrees")
        area = (e-w)*111.32*math.cos(math.radians((s+n)/2))*(n-s)*111.32
        if not .05 <= area <= 150:
            raise ValueError("AOI area must be approximately 0.05 to 150 square km")
        if not 1 <= (self.end_date-self.start_date).days <= 45:
            raise ValueError("Date range must be 1 to 45 days; end is exclusive")
        if self.end_date > date.today():
            raise ValueError("Future dates are not accepted")
        return self

    def aoi(self):
        w,s,e,n = self.bbox
        return {"type":"Polygon", "coordinates":[[[w,s],[e,s],[e,n],[w,n],[w,s]]]}

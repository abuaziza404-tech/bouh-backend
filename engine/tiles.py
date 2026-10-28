"""XYZ tile reprojection. Categorical classes always use nearest-neighbor."""
import io
import numpy as np
from PIL import Image
from .science import xyz_bounds,colorize


def tile_png(raster,z,x,y):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
    bounds=xyz_bounds(z,x,y);grid=from_bounds(*bounds,256,256)
    classes=np.full((256,256),255,dtype='uint8')
    with rasterio.open(raster) as src:
        reproject(source=rasterio.band(src,1),destination=classes,src_transform=src.transform,
                  src_crs=src.crs,src_nodata=255,dst_transform=grid,dst_crs='EPSG:3857',
                  dst_nodata=255,resampling=Resampling.nearest)
    buffer=io.BytesIO();Image.fromarray(colorize(classes)).save(buffer,format='PNG')
    return buffer.getvalue()

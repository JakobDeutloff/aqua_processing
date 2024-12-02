import os
import xarray as xr
import numpy as np

def get_grid(ds):
    uri = ds.grid_file_uri
    downloads_prefix = "http://icon-downloads.mpimet.mpg.de/grids/public/"
    if uri.startswith(downloads_prefix):
        local_grid_path = os.path.join(
            "/pool/data/ICON/grids/public", uri[len(downloads_prefix) :]
        )
    else:
        raise NotImplementedError(f"no idea about how to get {uri}")
    return xr.open_dataset(local_grid_path)


def merge_grid(data_ds):
    grid = get_grid(data_ds)
    ds = xr.merge([data_ds, grid[list(grid.coords)].rename({"cell": "ncells"})])
    ds = ds.assign(clon=np.degrees(ds.clon), clat=np.degrees(ds.clat))
    return ds
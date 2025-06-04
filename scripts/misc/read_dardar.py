# %%
import os
import gzip
import shutil
import tempfile
import xarray as xr
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# %%
path = '/work/um0878/data/dardar_subsel/2008/06/*/*.gz'
files = sorted(glob.glob(path))


# %%
def decompress_gz(gz_file, tmpdir):
    nc_file = os.path.join(tmpdir, os.path.basename(gz_file)[:-3])
    try:
        with gzip.open(gz_file, 'rb') as f_in, open(nc_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return nc_file
    except EOFError:
        print(f"Corrupted file skipped: {gz_file}")
        return None

with tempfile.TemporaryDirectory(dir='/work/bm1183/m301049') as tmpdir:
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda f: decompress_gz(f, tmpdir), files), total=len(files)))
    nc_files = [f for f in results if f is not None]
    ds = xr.open_mfdataset(nc_files, combine='nested', concat_dim='Collocations')
# %%

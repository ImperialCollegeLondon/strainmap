import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from strainmap.models.writers import save_group, write_netcdf_file, repack_file


ds = xr.Dataset(
    {
        "foo": (("x", "y"), np.random.rand(4, 5).astype(np.float32)),
        "bar": (("x"), np.random.rand(4).astype(np.float32)),
    },
    coords={
        "x": [10, 20, 30, 40],
        "y": pd.date_range("2000-01-01", periods=5),
        "z": ("y", list("abcde")),
    },
)

foo = ds.foo
bar = ds.bar

filename = Path(__file__).parent / "using_strainmap.nc"

# save_group(filename, foo, foo.name, to_int=True)
# save_group(filename, bar, bar.name, to_int=True)

# bar = bar.combine_first(xr.DataArray([3], dims=["x"], coords={"x": [50]}))
# save_group(filename, bar, bar.name, to_int=True)

write_netcdf_file(filename, foo=foo, bar=bar, author="Diego", age=38)

repack_file(filename, overwrite=True)

new_bar = xr.open_dataarray(filename, group=bar.name)
xr.testing.assert_identical(bar, new_bar)

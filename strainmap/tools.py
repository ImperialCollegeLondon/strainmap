import re
import sparse as sp
import xarray as xr


def get_sa_location(dataset):
    pattern = r"[sS][aA][xX]?([0-9])"
    m = re.search(pattern, dataset)
    return int(m.group(1)) if hasattr(m, "group") else 99


def to_sparse(array: xr.DataArray, fill_value=None) -> xr.DataArray:
    """ Transforms a dense xarray into an sparse.COO xarray.

    Args:
        array (xr.DataArray): Array to make sparse.
        fill_value (scalar): Fill value. Defaults to 0.

    Returns:
        An xarray with the same dimensions and coordinates but sparse.
    """
    return xr.DataArray(
        sp.COO.from_numpy(array.data, fill_value=fill_value),
        dims=array.dims,
        coords=array.coords,
        attrs=array.attrs
    )


def to_dense(array: xr.DataArray) -> xr.DataArray:
    """ Transforms a sparse.COO xarray into an dense numpy xarray.

    Args:
        array (xr.DataArray): Array to make dense.

    Returns:
        An xarray with the same dimensions and coordinates but dense.
    """
    return xr.DataArray(
        array.data.todense(),
        dims=array.dims,
        coords=array.coords,
        attrs=array.attrs
    )
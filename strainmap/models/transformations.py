from itertools import product
from typing import List

import numpy as np
import xarray as xr

from ..coordinates import Comp


def dict_to_xarray(structure: dict, dims: List[str]) -> xr.DataArray:
    """Convert a nested dictionary into a DataArray with the given dimensions.

    The dimensions are taken assumed ot be in the order they are provided, with the
    first dimension given by the outmost dictionary and the last dimension given by the
    innermost dictionary or numpy array.

    Args:
        structure:
        dims:

    Raises:
        KeyError: If there is any inconsistency in the number or type of elements at any
         level of the nested dictionary.
        IndexError: If there is any inconsistency in the shape of numpy arrays stored
         in the dictionaries.

    Returns:
        A DataArray with the dimensions indicated in the input, the coordinates as taken
        from the dictionary keys and, when there are numpy arrays, integer numerical
        sequences.
    """
    data = []
    for k, v in structure.items():
        if isinstance(v, dict):
            data.append(dict_to_xarray(v, dims[1:]).expand_dims(dim={dims[0]: [k]}))
        else:
            da = xr.DataArray(
                v,
                dims=dims[1:],
                coords={d: np.arange(0, s) for d, s in zip(dims[1:], v.shape)},
            )
            data.append(da.expand_dims(dim={dims[0]: [k]}))

    return xr.concat(data, dim=dims[0])


def masked_reduction(
    data: xr.DataArray, radial: xr.DataArray, angular: xr.DataArray
) -> xr.DataArray:
    """Reduces the size of an array by averaging the regions defined by the masks.

    The radial and angular masks define a collection of regions of interest in 2D space
    all together forming a torus, with Na angular segments and Nr radial segments.
    This means that a large 2D array with data (it can have more dimensions) can be
    reduced to a much smaller and easy to handle (Nr, Na) array, where the value of
    each entry is the mean value of the pixels in the regions defined by both masks.

    The reduced array has the dimensions of the input data with 'row' and 'col' removed
    and 'radius' and 'angle' added to the end. So if data shape is (N0, N1, N2, N3, N4)
    then the reduced array will have shape (N0, N3, N4, Nr, Na).

    Simplified example: If we had 2 radial and 4 angular regions, (8 in total) regions
    in an otherwise 8x8 array, the reduced array will be a 2x4 array, with each element
    the mean value of all the elements in that region:

    Regions:
    [[0, 0, 0, 0, 0, 0, 0, 0],
     [0, 5, 5, 5, 6, 6, 6, 0],
     [0, 5, 1, 1, 2, 2, 6, 0],
     [0, 5, 1, 0, 0, 2, 6, 0],
     [0, 8, 4, 0, 0, 3, 7, 0],
     [0, 8, 4, 4, 3, 3, 7, 0],
     [0, 8, 8, 8, 7, 7, 7, 0],
     [0, 0, 0, 0, 0, 0, 0, 0]]

     Reduced array, with the numbers indicating the region they related to:
     [[1, 2, 3, 4],
      [5, 6, 7, 8]]

    Args:
        data: Large array to reduce. Must contain all the dimensions of the masks
            except 'region'.
        radial: Radial mask. Must contain 'region', 'row' and 'col' dimensions, at
            least.
        angular: Angular mask. Must have same dimensions (and shape) that the radial
            mask.

    Returns:
        A DataArray with reduced size.
    """

    dims = [d for d in data.dims if d not in ("row", "col")]
    nrad = radial.sizes["region"]
    nang = angular.sizes["region"]
    reduced = xr.DataArray(
        np.zeros([data.sizes[d] for d in dims] + [nrad, nang], dtype=data.dtype),
        dims=dims + ["radius", "angle"],
        coords={d: data[d] for d in dims},
    )

    for r, a in product(range(nrad), range(nang)):
        mask = xr.apply_ufunc(
            np.logical_and, radial.isel(region=r), angular.isel(region=a)
        )
        reduced.loc[{"radius": r, "angle": a}] = (
            data.sel(row=mask.row, col=mask.col).where(mask).mean(dim=("row", "col"))
        )

    return reduced


def masked_expansion(
    data: xr.DataArray,
    radial: xr.DataArray,
    angular: xr.DataArray,
    nrow: int = 512,
    ncol: int = 512,
) -> xr.DataArray:
    """Transforms a reduced array into a full array with the same shape as the masks.

    This function, partially opposite to `masked_reduction`, will recover a full size
    array with the same shape as the original one and with the masked elements equal to
    the corresponding entries of the reduced array. All other elements are nan.

    Args:
        data: Reduced array to expand. Should include dimensions 'radius' and 'angle'
            as well as any dimension of the masks other than 'region', 'row' and
            'column'.
        radial: Radial mask. Must contain 'region', 'row' and 'col' dimensions, at
            least.
        angular: Angular mask. Must have same dimensions (and shape) that the radial
            mask.
        nrow: Size of 'row' dimension in the expanded array iof larger than the one of
            the radial array.
        ncol: Size of 'col' dimension in the expanded array iof larger than the one of
            the radial array.

    Returns:
        The expanded array.
    """
    dims = [d for d in data.dims if d not in ("radius", "angle")]
    nrad = data.sizes["radius"]
    nang = data.sizes["angle"]
    expanded = xr.DataArray(
        np.full(
            [data.sizes[d] for d in dims] + [radial.sizes["row"], radial.sizes["col"]],
            np.nan,
            dtype=data.dtype,
        ),
        dims=dims + ["row", "col"],
        coords={**{d: data[d] for d in dims}, "row": radial.row, "col": radial.col},
    )

    for r, a in product(range(nrad), range(nang)):
        mask = xr.apply_ufunc(
            np.logical_and, radial.sel(region=r), angular.sel(region=a)
        )
        expanded = xr.where(mask, data.sel(radius=r, angle=a), expanded)

    # Now we ensure that the dimensions are in the correct order
    expanded = expanded.transpose(*tuple(dims + ["row", "col"]))

    # Finally, we populate an array with the correct number of rows and cols, if needed
    if nrow != radial.sizes["row"] or ncol != radial.sizes["col"]:
        output = xr.DataArray(
            np.full(
                [data.sizes[d] for d in dims] + [nrow, ncol], np.nan, dtype=data.dtype
            ),
            dims=dims + ["row", "col"],
            coords={
                **{d: data[d] for d in dims},
                "row": np.arange(0, nrow),
                "col": np.arange(0, ncol),
            },
        )
        output.loc[{"row": expanded.row, "col": expanded.col}] = expanded

        return output
    else:
        return expanded


def coordinates(
    centroid: xr.DataArray,
    septum: xr.DataArray,
    radial: xr.DataArray,
    angular: xr.DataArray,
    pixel_size: float,
) -> xr.DataArray:
    """Produces a reduced array with the average location in the plane of the pixels.

    First, the location of each pixel of one of the masks is calculated in cylindrical
    coordinates as a function of the frame. Then, the array is masked-reduced to get
    the average value of those coordinates in the relevant superpixels.

    Args:
        centroid: Location of the centroids
        septum: Location of the septum midpoints.
        radial: Radial mask. Must contain 'region', 'row' and 'col' dimensions, at
            least.
        angular: Angular mask. Must have same dimensions (and shape) that the radial
            mask.
        pixel_size: physical size of the pixels

    Returns:
        The reduced array with the locations. The dimension `comp` will have
        coordinates `RAD` and `CIRC`, indicating the average superpixel position
        in the radial and the circumferential directions.
    """
    iframe, irow, icol = np.nonzero(radial.sum("region").data)

    crow = centroid.sel(frame=iframe, coord="row").data
    ccol = centroid.sel(frame=iframe, coord="col").data
    th0 = 2 * np.pi - theta_origin(centroid, septum).sel(frame=iframe).data

    row = radial.row[irow].data - crow
    col = radial.col[icol].data - ccol

    loc = xr.full_like(
        radial.isel(region=0).expand_dims(comp=[Comp.RAD.name, Comp.CIRC.name]),
        fill_value=np.nan,
        dtype="float",
    ).transpose("comp", "frame", "row", "col")
    loc.data[0, iframe, irow, icol] = np.sqrt(row**2 + col**2) * pixel_size
    loc.data[1, iframe, irow, icol] = np.mod(np.arctan2(col, row) - th0, 2 * np.pi)

    return masked_reduction(loc, radial, angular)


def theta_origin(centroid: xr.DataArray, septum: xr.DataArray):
    """Finds theta0 out of the centroid and septum mid-point"""
    shifted = septum - centroid
    theta0 = np.arctan2(shifted.sel(coord="col"), shifted.sel(coord="row"))
    return theta0

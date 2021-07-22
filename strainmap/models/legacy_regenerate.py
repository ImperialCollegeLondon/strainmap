from typing import List
import logging

import xarray as xr
import numpy as np

from .strainmap_data_model import StrainMapData
from .segmentation import _calc_centroids, _calc_effective_centroids
from .velocities import calculate_velocities
from ..coordinates import Comp


def _dict_to_xarray(structure: dict, dims: List[str]) -> xr.DataArray:
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
    coords = []
    for k, v in structure.items():
        coords.append(k)
        if isinstance(v, dict):
            data.append(_dict_to_xarray(v, dims[1:]).expand_dims(dim={dims[0]: [k]}))
        else:
            da = xr.DataArray(
                v,
                dims=dims[1:],
                coords={d: np.arange(0, s) for d, s in zip(dims[1:], v.shape)},
            )
            data.append(da.expand_dims(dim={dims[0]: [k]}))

    return xr.concat(data, dim=dims[0])


def regenerate_segmentation(structure: dict) -> dict:
    """

    Args:
        structure:

    Returns:

    """
    corrected = dict()

    if "orientation" in structure:
        corrected["orientation"] = "CCW" if "CCW" in structure["orientation"] else "CW"

    if "timeshift" in structure:
        corrected["timeshift"] = structure["timeshift"]

    sign = structure["sign_reversal"][1:] + structure["sign_reversal"][:1]
    corrected["sign_reversal"] = xr.DataArray(
        [1 if s == 1 else -1 for s in sign],
        dims=["comp"],
        coords={"comp": [Comp.X.name, Comp.Y.name, Comp.Z.name]},
        name="sign_reversal",
    ).astype(np.int16)

    corrected["segments"] = (
        _dict_to_xarray(
            structure["segments"], ["cine", "side", "frame", "coord", "point"]
        )
        .assign_coords({"coord": ["col", "row"]})
        .rename("segments")
    )

    raw_centroids = _calc_centroids(corrected["segments"])
    corrected["centroid"] = _calc_effective_centroids(raw_centroids, window=3).rename(
        "centroid"
    )

    corrected["septum"] = (
        _dict_to_xarray(
            {k: v[..., 0] for k, v in structure["zero_angle"].items()},
            ["cine", "frame", "coord"],
        )
        .assign_coords({"coord": ["col", "row"]})
        .rename("septum")
    )

    return corrected


def regenerate_velocities(data: StrainMapData) -> None:
    """Calculate the velocities for all of the segmentations available.

    Args:
        data: A StrainMap data object with segmentations
    """
    cines = data.segments.cine
    for i, cine in enumerate(data.segments.cine):
        logging.info(
            f"({i+1}/{cines.size}) Calculating velocities for cine {cine.item()}"
        )
        calculate_velocities(data, cine.item(), tuple(data.sign_reversal))

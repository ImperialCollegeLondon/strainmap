from typing import List
import logging
from pathlib import Path

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
    for k, v in structure.items():
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


def _regenerate_auxiliar(structure: dict) -> dict:
    """Transform auxiliary data stored in the h5 structure to the correct format.

    In particular, it deals with the following (optional) keys:
        - orientation
        - timeshift
        - sign_reversal

    Args:
        structure: Nested dictionary structure containing the relevant data.

    Returns:
        A dictionary with the above keys in the correct format.
    """
    corrected = dict()

    if "orientation" in structure:
        corrected["orientation"] = "CCW" if "CCW" in structure["orientation"] else "CW"

    if "timeshift" in structure:
        corrected["timeshift"] = structure["timeshift"]

    if "sign_reversal" in structure:
        sign = structure["sign_reversal"][1:] + structure["sign_reversal"][:1]
        corrected["sign_reversal"] = xr.DataArray(
            [1 if s == 1 else -1 for s in sign],
            dims=["comp"],
            coords={"comp": [Comp.X.name, Comp.Y.name, Comp.Z.name]},
            name="sign_reversal",
        ).astype(np.int16)

    return corrected


def _regenerate_segmentation(structure: dict) -> dict:
    """Regenerate the segmentation data as DataArrays.

    In particular, it requires the following keys:
        - segments
        - zero_angle

    Args:
        structure: Nested dictionary structure containing the relevant data.

    Returns:
        A Dictionary of DataArray containing the 'segments', 'centroid' and 'septum'
    """
    corrected = dict()

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


def _regenerate_velocities(data: StrainMapData) -> None:
    """Calculate the velocities for all of the segmentations available.

    It updates the input data object with the new information.

    Args:
        data: A StrainMap data object with segmentations
    """
    cines = data.segments.cine
    for i, cine in enumerate(data.segments.cine):
        logging.info(
            f"({i+1}/{cines.size}) Calculating velocities for cine {cine.item()}"
        )
        calculate_velocities(data, cine.item(), tuple(data.sign_reversal))


def regenerate(data: StrainMapData, structure: dict, filename: Path) -> None:
    """Regenerate a StrainMap data object from the data stored in the legacy h5 format.

    It updates the input data object with the new information.

    Args:
        data: A StrainMap data object
        structure: Nested dictionary structure containing the relevant data.
        filename: The name of the h5 file
    """
    data.__dict__.update(_regenerate_auxiliar(structure))
    data.__dict__.update(_regenerate_segmentation(structure))
    _regenerate_velocities(data)
    data.add_file(filename.with_suffix(".nc"))

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from ..coordinates import Comp
from .segmentation import _calc_centroids, _calc_effective_centroids
from .strainmap_data_model import StrainMapData
from .transformations import dict_to_xarray
from .velocities import calculate_velocities


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
        # sign = structure["sign_reversal"][1:] + structure["sign_reversal"][:1]
        corrected["sign_reversal"] = xr.DataArray(
            [-1 if s else 1 for s in structure["sign_reversal"]],
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
        dict_to_xarray(
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
        dict_to_xarray(
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

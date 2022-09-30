from typing import Callable, Dict, Tuple, Union

import numpy as np
import xarray as xr
from scipy import ndimage

from .ai_segmenter import ai_segmentation
from .snakes_segmenter import snakes_segmentation
from .strainmap_data_model import StrainMapData

SEGMENTATION_METHOD: Dict[str, Callable] = {
    "snakes": snakes_segmentation,
    "ai": ai_segmentation,
}


def new_segmentation(
    data: StrainMapData,
    cine: str,
    frame: Union[int, None],
    initials: xr.DataArray,
    new_septum: xr.DataArray,
    method: str = "snakes",
) -> None:
    """Starts a new segmentation, either quick or step by step.

    If it is a quick one, after calculating the segmentation, the effective centroids
    are also calculated.
    """
    segments, centroid, septum = _get_segment_variables(
        data, cine, initials.sizes["point"]
    )
    frm = [frame] if isinstance(frame, int) else segments.frame.data.tolist()

    segments.loc[{"frame": frm}] = SEGMENTATION_METHOD.get(method, snakes_segmentation)(
        data.data_files.images(cine).sel(frame=frm),
        initials=initials,
        points=initials.sizes["point"],
    )

    # Update the centroid (or center of mass, COM)
    centroid.loc[{"frame": frm}] = _calc_centroids(segments.sel(frame=frm))

    # We set the new septum
    septum.loc[{"frame": frm}] = new_septum.sel(frame=0).copy()

    # If are doing an automatic segmentation for all frames, calculate the effective COM
    if len(frm) == len(segments.frame):
        centroid[...] = _calc_effective_centroids(centroid, window=3)
        _reset_stale_vel_and_strain(data, cine)
        data.save_all()
    else:
        data.save("segments", "centroid", "septum")


def update_segmentation(
    data: StrainMapData, cine: str, new_segments: xr.DataArray, new_septum: xr.DataArray
) -> None:
    """Updates the segmentation for the current frame.

    If there are no NaNs in the segments array, then we update the effective centroids
    as the segmentation is complete. Updating a complete segmentation deletes any
    derived quantity, like velocities, markers and all the strain results.
    """
    segments, centroid, septum = _get_segment_variables(data, cine)
    frame = new_segments.frame

    # Update the segment with the new one
    segments.loc[{"frame": frame}] = new_segments.copy()

    # Update the centroid (or center of mass, COM)
    centroid.loc[{"frame": frame}] = _calc_centroids(segments.sel(frame=frame))

    # Set the new septum
    septum.loc[{"frame": frame}] = new_septum.sel(frame=frame).copy()

    if not xr.apply_ufunc(np.isnan, segments).any():
        centroid[...] = _calc_effective_centroids(centroid, window=3)
        _reset_stale_vel_and_strain(data, cine)
        data.save_all()
    else:
        data.save("segments", "centroid", "septum")


def update_and_find_next(
    data: StrainMapData,
    cine: str,
    frame: int,
    new_segments: xr.DataArray,
    new_septum: xr.DataArray,
    method: str = "snakes",
) -> None:
    """Updates the segmentation for the current frame and starts the next one.

    The segmentation of the next frame uses the previous one as initial value and takes
    the same septum.
    """

    update_segmentation(data, cine, new_segments, new_septum)

    segments, centroid, septum = _get_segment_variables(data, cine)

    segments.loc[{"frame": frame}] = SEGMENTATION_METHOD.get(
        method, snakes_segmentation
    )(
        data.data_files.images(cine).sel(frame=[frame]),
        initials=segments.sel(frame=frame - 1),
        points=segments.sizes["point"],
    ).squeeze(
        "frame", drop=True
    )

    # We update the centroid (or center of mass, COM)
    centroid.loc[{"frame": frame}] = _calc_centroids(segments.sel(frame=frame))

    # We set the new septum
    septum.loc[{"frame": frame}] = new_septum.sel(frame=frame - 1).copy()


def remove_segmentation(data: StrainMapData, cine: str) -> None:
    """Removes the segmentation for the given cine.

    It also removes all the derived magnitudes form that segmentation. In the case of
    strain, it removes all the strain related data from all cines as the calculated
    strain is no longer valid if one of the segmentations it depends on is changed
    or removed.
    """
    data.segments = _drop_cine(data.segments, cine)
    data.centroid = _drop_cine(data.centroid, cine)
    data.septum = _drop_cine(data.septum, cine)
    _reset_stale_vel_and_strain(data, cine)
    data.save_all()


def _reset_stale_vel_and_strain(data: StrainMapData, cine: str) -> None:
    """Removes the cine from the velocity arrays and resets the strain.

    Args:
        data: A StrainMap data object.
        cine: The cine to remove.

    Returns:
        None
    """
    data.masks = _drop_cine(data.masks, cine)
    data.cylindrical = _drop_cine(data.cylindrical, cine)
    data.velocities = _drop_cine(data.velocities, cine)
    data.markers = _drop_cine(data.markers, cine)

    data.strain = xr.DataArray()
    data.strain_markers = xr.DataArray()


def _drop_cine(x: xr.DataArray, cine: str) -> xr.DataArray:
    """Removes a cine from the dataframe."""
    if not hasattr(x, "cine"):
        return x
    out = x.where(x.cine != cine, drop=True)
    return out if out.size > 0 else xr.DataArray()


def _get_segment_variables(
    data: StrainMapData, cine: str, points: int = 360
) -> Tuple[xr.DataArray, ...]:
    """Get the relevant segmentation variables for this cine.

    If they do not exist already in the data structure, they are created.

    Args:
        data (StrainMapData): The data object containing all the information.
        cine (str): Cine of interest.
        points (int): Number of points in a segment.

    Returns:
        A tuple with the segment, centroid and septum for the chosen cine.
    """
    frames = data.data_files.frames

    if data.segments.shape == ():
        data.segments = _init_segments(cine, frames, points)
        data.centroid = _init_septum_and_centroid(cine, frames, "centroid")
        data.septum = _init_septum_and_centroid(cine, frames, "septum")

    elif cine not in data.segments.cine:
        segments = _init_segments(cine, frames, points)
        data.segments = xr.concat([data.segments, segments], dim="cine")
        centroid = _init_septum_and_centroid(cine, frames, "centroid")
        data.centroid = xr.concat([data.centroid, centroid], dim="cine")
        septum = _init_septum_and_centroid(cine, frames, "septum")
        data.septum = xr.concat([data.septum, septum], dim="cine")

    return (
        data.segments.sel(cine=cine),
        data.centroid.sel(cine=cine),
        data.septum.sel(cine=cine),
    )


def _init_segments(cine: str, frames: int, points: int) -> xr.DataArray:
    """Initialises an empty segments DataArray."""
    return xr.DataArray(
        np.full((1, 2, frames, 2, points), np.nan),
        dims=("cine", "side", "frame", "coord", "point"),
        coords={
            "cine": [cine],
            "side": ["endocardium", "epicardium"],
            "coord": ["col", "row"],
            "frame": np.arange(0, frames),
        },
        name="segments",
    )


def _init_septum_and_centroid(cine: str, frames: int, name: str) -> xr.DataArray:
    """Initialises an empty septum mid point DataArray."""
    return xr.DataArray(
        np.full((1, frames, 2), np.nan),
        dims=("cine", "frame", "coord"),
        coords={"cine": [cine], "coord": ["col", "row"], "frame": np.arange(0, frames)},
        name=name,
    )


def _calc_centroids(segments: xr.DataArray) -> xr.DataArray:
    """Return an array with the position of the centroid at a given time."""
    return segments.sel(side="epicardium").mean(dim="point").drop("side")


def _calc_effective_centroids(centroid: xr.DataArray, window: int = 3) -> xr.DataArray:
    """Returns the rolling average of the centroid for the chosen window.

    The rolling average is wrapped at the edges of the array.

    Args:
        centroid (xr.DataArray): Array with the current centroids.
        window (int): Width of the window at each side. eg, 3 is rolling average of ±3

    Returns:
        A DataArray with the new centroid positions.
    """
    axis = centroid.dims.index("frame")
    weights = np.full((2 * window + 1), 1 / (2 * window + 1))
    return xr.apply_ufunc(
        lambda x: ndimage.convolve1d(x, weights, axis=axis, mode="wrap"), centroid
    )

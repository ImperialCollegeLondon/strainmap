from copy import copy
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import xarray as xr
from scipy import ndimage

from ..coordinates import Comp
from .contour_mask import Contour, dilate
from .segmenters import Segmenter
from .strainmap_data_model import StrainMapData

model = "AC"
model_params = {
    "endocardium": dict(alpha=0.0098, beta=15.5, gamma=0.0022),
    "epicardium": dict(alpha=0.0098, beta=15.5, gamma=0.0022),
}

ffilter = "gaussian"
filter_params: Dict[str, Dict] = {
    "endocardium": dict(sigma=2),
    "epicardium": dict(sigma=2),
}

propagator = "initial"
propagator_params: Dict[str, Dict] = {"endocardium": dict(), "epicardium": dict()}

rtol_endo: float = 0.15
rtol_epi: float = 0.10


def new_segmentation(
    data: StrainMapData,
    cine: str,
    frame: Union[int, None],
    initials: xr.DataArray,
    new_septum: xr.DataArray,
    replace_threshold: int = 31,
) -> None:
    """Starts a new segmentation, either quick or step by step.

    If it is a quick one, after calculating the segmentation, the effective centroids
    are also calculated.
    """
    segments, centroid, septum = _get_segment_variables(
        data, cine, initials.sizes["point"]
    )
    frame = frame if frame is not None else segments.frame

    _find_segmentation(
        segments,
        centroid,
        data.data_files.images(cine).sel(frame=frame, comp=Comp.MAG),
        initials,
        replace_threshold,
    )

    # If are doing an automatic segmentation for all frames, calculate the effective COM
    if (frame == segments.frame).all():
        centroid[...] = _calc_effective_centroids(centroid, window=3)

    # We set the new septum
    septum.loc[{"frame": frame}] = new_septum.sel(frame=0).copy()

    # data.save("segments", "centroid", "septum")


def update_segmentation(
    data: StrainMapData, cine: str, new_segments: xr.DataArray, new_septum: xr.DataArray
) -> None:
    """Updates the segmentation for the current frame.

    If there're no NaNs in the segments array, then we update the effective centroids as
    the segmentation is complete.Updating a complete segmentation deletes any derived
    quantity, like velocities, markers and all the strain results.
    """
    segments, centroid, septum = _get_segment_variables(data, cine)

    _update_segmentation(segments, centroid, new_segments)

    septum[...] = new_septum.copy()

    if not xr.ufuncs.isnan(segments).any():
        raw_centroids = _calc_centroids(segments)
        centroid[...] = _calc_effective_centroids(raw_centroids, window=3)

        # data.velocities.pop(cine, None)
        # data.masks.pop(cine, None)
        # data.markers.pop(cine, None)
        # data.strain = xr.DataArray()
        # data.strain_markers = xr.DataArray()

        # data.save_all()

    # else:
    #     data.save("segments", "centroid", "septum")


def update_and_find_next(
    data: StrainMapData,
    cine: str,
    frame: int,
    new_segments: xr.DataArray,
    new_septum: xr.DataArray,
    replace_threshold: int = 31,
) -> None:
    """Updates the segmentation for the current frame and starts the next one.

    The segmentation of the next frame uses the previous one as initial value and takes
    the same septum.
    """

    update_segmentation(data, cine, new_segments, new_septum)

    segments, centroid, septum = _get_segment_variables(data, cine)
    _find_segmentation(
        segments,
        centroid,
        data.data_files.images(cine).sel(frame=frame, comp=Comp.MAG),
        segments.sel(frame=frame - 1),
        replace_threshold,
    )

    # We set the new septum
    septum.loc[{"frame": frame}] = new_septum.sel(frame=frame - 1).copy()


def remove_segmentation(data: StrainMapData, cine: str) -> None:
    """Removes the segmentation for the given cine.

    It also removes all the derived magnitudes form that segmentation. In the case of
    strain, it removes all the strain related data from all cines as the calculated
    strain is no longer valid if one of the segmentations it depends on is changed
    or removed.
    """
    data.segments = data.segments.where(data.segments.cine != cine, drop=True)
    data.centroid = data.septum.where(data.centroid.cine != cine, drop=True)
    data.septum = data.septum.where(data.septum.cine != cine, drop=True)
    data.velocities.pop(cine, None)
    data.masks.pop(cine, None)
    data.markers.pop(cine, None)
    data.strain = xr.DataArray()
    data.strain_markers = xr.DataArray()

    # data.save_all()


def _get_segment_variables(
    data: StrainMapData, cine: str, points: int = 360
) -> Tuple[xr.DataArray, ...]:
    """Get the relevant segementation variables for this cine.

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


def _find_segmentation(
    segments: xr.DataArray,
    centroid: xr.DataArray,
    images: xr.DataArray,
    initials: xr.DataArray,
    replace_threshold: int = 31,
) -> None:
    """Find the segmentation of one or more images starting at the initials segments.

    Args:
        segments (xr.DataArray): The original segments array.
        centroid (xr.DataArray): The original centroids array.
        images (xr.DataArray): The images for the frames to segment.
        initials (xr.DataArray): Initial segments to start the segmentation with.

    Returns:
        None
    """
    frame = images.frame

    rules = (
        _create_rules_one_frame(
            frame.item(), segments, (images.sizes["row"], images.sizes["col"])
        )
        if frame.shape == ()
        else _create_rules_all_frames(replace_threshold)
    )

    for side in ("endocardium", "epicardium"):
        # In frame by frame segmentation, we don't segment if already above threshold
        if frame.shape == () and frame.item() >= replace_threshold:
            segments.loc[{"side": side, "frame": frame}] = segments.sel(
                side=side, frame=frame - 1
            ).copy()

        # For multiple segmentations or if not yet above threshold
        else:
            segments.loc[{"side": side, "frame": frame}] = _simple_segmentation(
                images.data,
                initials.sel(side=side).data,
                model=model,
                model_params=model_params[side],
                ffilter=ffilter,
                filter_params=filter_params[side],
                propagator=propagator,
                propagator_params=propagator_params[side],
                rules=rules[side],
            )

    # We update the centroid (or center of mass, COM)
    centroid.loc[{"frame": frame}] = _calc_centroids(segments.sel(frame=frame))


def _update_segmentation(
    segments: xr.DataArray,
    centroid: xr.DataArray,
    new_segments: Optional[xr.DataArray] = None,
) -> None:
    """ Updates an existing segmentation and septum with new ones.

    Any velocities calculated for this cine are deleted, forcing their recalculation.

    Args:
        segments (xr.DataArray): The original segments array.
        centroid (xr.DataArray): The original centroids array.
        new_segments (optional, xr.DataArray): The new segments.

    Returns:
        None
    """
    segments.loc[{"frame": new_segments.frame}] = new_segments.copy()
    centroid.loc[{"frame": new_segments.frame}] = _calc_centroids(new_segments)


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


def _create_rules_one_frame(
    frame: xr.DataArray, replacement: xr.DataArray, shape: Tuple[int, int]
):
    """Create the rules to apply to the segments to ensure their 'quality'.

    - Expand/contract the contour
    - Replace for the previous segmentation is area change above tolerance.

    Args:
        frame (int): Frame at which to apply the rules.
        segments(xr.DataArray): Array with all the contours
        shape (tuple): Shape of the image. Needed to expand/contract the contour
            (well, not really)

    Returns:
        A dictionary with the set of rules for the endo and epicardium contours
    """
    rules = {"endocardium": [], "epicardium": []}
    shift = {"endocardium": 2, "epicardium": -2}
    rtol = {"endocardium": rtol_endo, "epicardium": rtol_epi}

    for side in ("endocardium", "epicardium"):
        rules[side].append(partial(dilate, s=shift[side]))
        if frame == 0:
            continue

        rules[side].append(
            partial(
                _replace_single,
                replacement=Contour(
                    replacement.sel(side=side, frame=frame - 1).data.T, shape=shape
                ),
                rtol=rtol[side],
                replace=False,
            )
        )

    return rules


def _create_rules_all_frames(replace_threshold: int = 31):
    """Create the rules to apply to the segments to ensure their 'quality'.

    - Expand/contract the contour
    - Replace contour by the previous one if area change is above tolerance or frame
        number is above threshold.

    Returns:
        A dictionary with the set of rules for the endo and epicardium contours
    """
    rules = {"endocardium": [], "epicardium": []}
    shift = {"endocardium": 2, "epicardium": -2}
    rtol = {"endocardium": rtol_endo, "epicardium": rtol_epi}

    for side in ("endocardium", "epicardium"):
        rules[side].append(lambda c: list(map(partial(dilate, s=shift[side]), c)))
        rules[side].append(
            partial(
                _replace_in_list, rtol=rtol[side], frame_threshold=replace_threshold
            )
        )
    return rules


def _simple_segmentation(
    data: np.ndarray,
    initial: np.ndarray,
    model: Text,
    ffilter: Text,
    propagator: Text,
    model_params: Dict[Text, Any],
    filter_params: Dict[Text, Any],
    propagator_params: Dict[Text, Any],
    rules: List[Callable[[Contour], Contour]],
) -> Union[np.ndarray, List[np.ndarray]]:
    """Performs a segmentation of the data with the chosen parameters.

    Args:
        data: 2D or 3D numpy array with the images. If 3D, time should be the 3rd axis
        initial: A 2D array with the XY coordinates of the initial contour.
        model: Segmentation model to use. Possible options are:
            - 'AC': To use Active contours. Does not support 3D segmentation.
            - 'MorphGAC': To use the morphological geodesic active contours model
            - 'MorphCV': To use the morphological Chan Vese model
        ffilter: Filter to use. Possible options are:
            - 'gaussian': To use a gaussian filter
            - 'inverse_gaussian': To use an inverse gaussian filter
        propagator: How to propagate the segmentation from one frame to the next.
            Possible options are:
            - None: A 3D segmentation is used, so no propagation is needed.
            - 'initial': Just use the same initial contour for all frames.
            - 'previous': Uses the segmentation of the previous frame as initial contour
                for the next one. Optionally, this can be expanded using a
                'dilation_factor', with values <1 shrinking the contour and values >1
                expanding it.
            - 'weighted': Uses a weighted average between the previous contour and the
                initial one. The relative weight is given by the keyword 'weight',
                with 1 resulting in the initial contour and 0 resulting in the previous
                one.
        model_params: Dictionary with the parameters to be passed to the model. Possible
            parameters are described in the corresponding model documentation.
        filter_params: Dictionary with the parameters to be passed to the filter.
            Possible parameters are described in the corresponding filter documentation.
        propagator_params: Dictionary with the parameters to be passed to the
            propagator. See the description of the propagators above.
        rules: List of rules (callables) to ensure the quality of the segmentation.

    Returns:
        A 2D or 3D numpy array with the coordinates of the contour resulting from the
        segmentation. If the data input is 3D, the returning array will also be 3D.
    """
    segmenter = Segmenter.setup(model=model, ffilter=ffilter, propagator=propagator)

    shape = data.shape[-2:]

    if initial.shape[1] != 2:
        initial = initial.T

    segmentation = segmenter(
        data,
        Contour(initial, shape=shape),
        model_params=model_params,
        filter_params=filter_params,
        propagator_params=propagator_params,
    )

    segmentation = reduce(lambda s, f: f(s), rules, segmentation)

    return (
        np.array([c.xy for c in segmentation]).transpose([0, 2, 1])
        if isinstance(segmentation, list)
        else segmentation.xy.T
    )


def _replace_single(
    contour: Contour, replacement: Contour, rtol: float = 0.15, replace: bool = True
) -> Contour:
    value = abs(contour.mask.sum() - replacement.mask.sum()) / replacement.mask.sum()
    if replace or value > rtol:
        return copy(replacement)

    return contour


def _replace_in_list(
    contour: List[Contour], rtol: float = 0.15, frame_threshold: int = 30
) -> List[Contour]:
    result = [contour[0]]
    for i, c in enumerate(contour[1:]):
        result.append(_replace_single(c, result[i], rtol, i + 1 >= frame_threshold))

    return result

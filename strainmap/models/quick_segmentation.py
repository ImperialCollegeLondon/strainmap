from .segmenters import Segmenter
from .strainmap_data_model import StrainMapData
from .contour_mask import Contour, dilate

import numpy as np
from copy import copy
from typing import Text, Dict, Any, Union, List, Callable
from functools import partial, reduce


def find_segmentation(
    data: StrainMapData,
    dataset_name: str,
    frame: Union[int, slice, None],
    images: Dict[str, np.ndarray],
    initials: Dict[str, np.ndarray],
    rtol: float = 0.15,
    replace_threshold: int = 30,
) -> StrainMapData:
    """Find the segmentation for the endocardium and the epicardium at one single frame.

    Args:
        data: StrainMapData object to store the result
        dataset_name: Name of the dataset to segment
        frame: index of the frame to segment
        images: Dictionary with the images to segment for the epi- and endocardium
        initials: Dictionary with the initial segmentation for the epi- and endocardium.
        replace_threshold: frame threshold from where endocardium segment must be
            replaced

    Returns:
        The StrainMapData object updated with the segmentation.
    """
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

    if len(data.segments[dataset_name].get("endocardium", [])) == 0:
        data = initialize_data_segments(
            data, dataset_name, initials["endocardium"].shape
        )

    rules = create_rules(
        frame,
        data.segments[dataset_name],
        images["endocardium"].shape[-2:],
        rtol,
        replace_threshold,
    )

    results = {}
    for side in ("endocardium", "epicardium"):
        if (
            side == "endocardium"
            and isinstance(frame, int)
            and frame >= replace_threshold
        ):
            results[side] = copy(data.segments[dataset_name][side][frame - 1])
        else:
            results[side] = simple_segmentation(
                images[side],
                initials[side],
                model=model,
                model_params=model_params[side],
                ffilter=ffilter,
                filter_params=filter_params[side],
                propagator=propagator,
                propagator_params=propagator_params[side],
                rules=rules[side],
            )

    if frame is None:
        frame = slice(None)
    elif isinstance(frame, int):
        frame = slice(frame, frame + 1)

    data.segments[dataset_name]["endocardium"][frame] = results["endocardium"]
    data.segments[dataset_name]["epicardium"][frame] = results["epicardium"]

    data.save(
        ["segments", dataset_name, "endocardium"],
        ["segments", dataset_name, "epicardium"],
        ["zero_angle", dataset_name],
    )

    return data


def initialize_data_segments(data, dataset_name, shape):
    """Initialises the StrainMap data object with empty segments."""
    if len(shape) == 2:
        num_frames = len(data.data_files[dataset_name]["MagZ"])
        shape = (num_frames,) + shape
    else:
        num_frames = shape[0]

    if shape[-1] == 2:
        shape = (shape[0], shape[2], shape[1])

    data.segments[dataset_name]["endocardium"] = np.full(shape, np.nan)
    data.segments[dataset_name]["epicardium"] = np.full(shape, np.nan)
    data.zero_angle[dataset_name] = np.full((num_frames, 2, 2), np.nan)

    return data


def create_rules(frame, segments, shape, rtol, replace_threshold):
    """Create the rules to apply to the segments to ensure their 'quality'."""
    rules = {"endocardium": [], "epicardium": []}
    threshold = {"endocardium": replace_threshold, "epicardium": np.inf}
    shift = {"endocardium": 1, "epicardium": -1}

    for side in ("endocardium", "epicardium"):
        if isinstance(frame, int):
            rules[side].append(partial(dilate, s=shift[side]))
            if frame > 0:
                rules[side].append(
                    partial(
                        replace_single,
                        replacement=Contour(segments[side][frame - 1].T, shape=shape),
                        rtol=rtol,
                        replace=False,
                    )
                )

        elif frame is None or isinstance(frame, slice):
            rules[side].append(lambda c: list(map(partial(dilate, s=shift[side]), c)))
            rules[side].append(
                partial(replace_in_list, rtol=rtol, frame_threshold=threshold[side])
            )

    return rules


def update_segmentation(
    data: StrainMapData,
    dataset_name: str,
    segments: dict,
    zero_angle: np.ndarray,
    frame: Union[int, slice],
) -> StrainMapData:
    """Updates an existing segmentation with new segments, potentially clearing them.

    Args:
        data: StrainMapData object containing the data
        dataset_name: Name of the dataset whose segmentation are to modify
        segments: Dictionary with the segmentation for the epi- and endocardium. If
            either is None, the existing segmentation is erased.
        frame: int or slice indicating the frames to update

    Returns:
        The StrainMapData object updated with the segmentation.
    """
    if segments["endocardium"] is not None and segments["epicardium"] is not None:
        data.segments[dataset_name]["endocardium"][frame] = copy(
            segments["endocardium"][frame]
        )
        data.segments[dataset_name]["epicardium"][frame] = copy(
            segments["epicardium"][frame]
        )
        data.zero_angle[dataset_name][frame] = copy(zero_angle[frame])

        data.save(
            ["segments", dataset_name, "endocardium"],
            ["segments", dataset_name, "epicardium"],
            ["zero_angle", dataset_name],
        )
    else:
        data.segments.pop(dataset_name, None)
        data.zero_angle.pop(dataset_name, None)

        data.delete(
            ["segments", dataset_name],
            ["zero_angle", dataset_name],
            ["velocities", dataset_name],
            ["masks", dataset_name],
            ["markers", dataset_name],
        )

    return data


def simple_segmentation(
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


def replace_single(
    contour: Contour, replacement: Contour, rtol: float = 0.15, replace: bool = True
) -> Contour:

    value = abs(contour.mask.sum() - replacement.mask.sum()) / replacement.mask.sum()
    if replace or value > rtol:
        return copy(replacement)

    return contour


def replace_in_list(
    contour: List[Contour], rtol: float = 0.15, frame_threshold: int = 30
) -> List[Contour]:

    result = [contour[0]]
    for i, c in enumerate(contour[1:]):
        result.append(replace_single(c, result[i], rtol, i + 1 >= frame_threshold))

    return result

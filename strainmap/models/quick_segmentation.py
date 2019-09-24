from .segmenters import Segmenter
from .strainmap_data_model import StrainMapData
from .contour_mask import Contour

import numpy as np
from copy import copy
from typing import Text, Dict, Any, Union, List


def find_segmentation(
    data: StrainMapData,
    dataset_name: str,
    frame: Union[int, slice, None],
    images: Dict[str, np.ndarray],
    initials: Dict[str, np.ndarray],
) -> StrainMapData:
    """Find the segmentation for the endocardium and the epicardium at one single frame.

    Args:
        data: StrainMapData object to store the result
        dataset_name: Name of the dataset to segment
        frame: index of the frame to segment
        images: Dictionary with the images to segment for the epi- and endocardium
        initials: Dictionary with the initial segmentation for the epi- and endocardium.

    Returns:
        The StrainMapData object updated with the segmentation.
    """
    model = "AC"
    model_params = {
        "endocardium": dict(alpha=0.01, beta=10, gamma=0.002),
        "epicardium": dict(alpha=0.01, beta=10, gamma=0.002),
    }

    ffilter = "gaussian"
    filter_params: Dict[str, Dict] = {"endocardium": dict(), "epicardium": dict()}

    propagator = "initial"
    propagator_params: Dict[str, Dict] = {"endocardium": dict(), "epicardium": dict()}

    if len(data.segments[dataset_name].get("endocardium", [])) == 0:
        data = initialize_data_segments(
            data, dataset_name, initials["endocardium"].shape
        )

    results = {}
    for side in ("endocardium", "epicardium"):
        results[side] = simple_segmentation(
            images[side],
            initials[side],
            model=model,
            model_params=model_params[side],
            ffilter=ffilter,
            filter_params=filter_params[side],
            propagator=propagator,
            propagator_params=propagator_params[side],
        )

    if frame is None:
        frame = slice(None)
    elif isinstance(frame, int):
        frame = slice(frame, frame + 1)
    print(frame)
    print(data.segments[dataset_name]["endocardium"].shape)
    data.segments[dataset_name]["endocardium"][frame] = results["endocardium"]
    data.segments[dataset_name]["epicardium"][frame] = results["epicardium"]

    return data


def initialize_data_segments(data, dataset_name, shape):
    """Initialises the StrainMap data object with empty segments."""
    if len(shape) == 2:
        num_frames = len(data.data_files[dataset_name]["MagZ"])
        shape = (num_frames,) + shape
    else:
        num_frames = shape[0]
    data.segments[dataset_name]["endocardium"] = np.full(shape, np.nan)
    data.segments[dataset_name]["epicardium"] = np.full(shape, np.nan)
    data.zero_angle[dataset_name] = np.full((num_frames, 2, 2), np.nan)

    return data


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
    else:
        data.segments.pop(dataset_name)
        data.zero_angle.pop(dataset_name)
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

    Returns:
        A 2D or 3D numpy array with the coordinates of the contour resulting from the
        segmentation. If the data input is 3D, the returning array will also be 3D.
    """
    segmenter = Segmenter.setup(model=model, ffilter=ffilter, propagator=propagator)

    shape = data.shape[:2]

    if initial.shape[1] != 2:
        initial = initial.T

    segmentation = segmenter(
        data,
        Contour(initial, shape=shape),
        model_params=model_params,
        filter_params=filter_params,
        propagator_params=propagator_params,
    )

    return (
        np.array([c.xy.T for c in segmentation]).transpose((0, 2, 1))
        if isinstance(segmentation, list)
        else segmentation.xy.T
    )

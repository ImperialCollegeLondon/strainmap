from .segmenters import Segmenter
from .strainmap_data_model import StrainMapData
from .contour_mask import Contour

import numpy as np
from typing import Text, Dict, Any


def find_segmentation(
    data: StrainMapData,
    dataset_name: str,
    phantom_dataset_name: str,
    targets: dict,
    initials: dict,
) -> StrainMapData:
    """Find the segmentation for the endocardium and the epicardium.

    Args:
        data: StrainMapData object containing the data
        dataset_name: Name of the dataset to segment
        phantom_dataset_name: Name of the dataset to use as phantom
        targets: Dictionary that relates endocardium/epicardium with the images to be
            used in that segmentation: mag or vel
        initials: Dictionary with the initial segmentation for the epi- and endcardioum.

    Returns:
        The StrainMapData object updated with the segmentation.
    """
    model = "AC"
    model_params_endo = dict(alpha=0.01, beta=10, gamma=0.002)
    model_params_epi = dict(alpha=0.01, beta=10, gamma=0.002)

    ffilter = "gaussian"
    filter_params_endo: dict = {}
    filter_params_epi: dict = {}

    propagator = "initial"
    propagator_params_endo: dict = {}
    propagator_params_epi: dict = {}

    segmenter = Segmenter.setup(model=model, ffilter=ffilter, propagator=propagator)

    all_data = get_data_to_segment(data, dataset_name, phantom_dataset_name)
    shape = all_data[targets["endocardium"]][0].shape

    data.segments[dataset_name]["endocardium"] = segmenter(
        all_data[targets["endocardium"]],
        Contour(initials["endocardium"].T, shape=shape),
        model_params=model_params_endo,
        filter_params=filter_params_endo,
        propagator_params=propagator_params_endo,
    )
    data.segments[dataset_name]["epicardium"] = segmenter(
        all_data[targets["epicardium"]],
        Contour(initials["epicardium"].T, shape=shape),
        model_params=model_params_epi,
        filter_params=filter_params_epi,
        propagator_params=propagator_params_epi,
    )

    return data


def get_data_to_segment(data, dataset_name, phantom_dataset_name):
    """Gets the data that will be segmented and removes the phantom, if needed"""
    magz = data.get_images(dataset_name, "MagZ")
    magx = data.get_images(dataset_name, "MagX")
    magy = data.get_images(dataset_name, "MagY")
    mag = magx + magy + magz
    vel = data.get_images(dataset_name, "PhaseZ")

    if phantom_dataset_name != "":
        magz = data.get_bg_images(phantom_dataset_name, "MagZ")
        magx = data.get_bg_images(phantom_dataset_name, "MagX")
        magy = data.get_bg_images(phantom_dataset_name, "MagY")
        mag = mag - (magx + magy + magz)
        vel = vel - data.get_bg_images(phantom_dataset_name, "PhaseZ")

    return {"mag": mag, "vel": vel}


def simple_segmentation(
    data: np.ndarray,
    initial: np.ndarray,
    model: Text,
    ffilter: Text,
    propagator: Text,
    model_params: Dict[Text, Any],
    filter_params: Dict[Text, Any],
    propagator_params: Dict[Text, Any],
) -> np.ndarray:
    """Performs a segmentatino of the data with the chosen parameters.

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

    segmentation = segmenter(
        data,
        Contour(initial, shape=shape),
        model_params=model_params,
        filter_params=filter_params,
        propagator_params=propagator_params,
    )

    return (
        np.array([c.xy for c in segmentation])
        if isinstance(segmentation, list)
        else segmentation.xy
    )

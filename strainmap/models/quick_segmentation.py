from .segmenters import Segmenter
from .strainmap_data_model import StrainMapData
from .contour_mask import Contour


def find_segmentation(
    data: StrainMapData, dataset_name: str, targets: dict, initials: dict
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

    all_data = get_data_to_segment(data, dataset_name)
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


def get_data_to_segment(data, dataset_name):
    """Gets the data that will be segmented and removes the phantom, if needed"""
    magz = data.get_images(dataset_name, "MagZ")
    magx = data.get_images(dataset_name, "MagX")
    magy = data.get_images(dataset_name, "MagY")
    mag = magx + magy + magz
    vel = data.get_images(dataset_name, "PhaseZ")

    return {"mag": mag, "vel": vel}

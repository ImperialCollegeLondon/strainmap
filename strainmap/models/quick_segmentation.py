from .segmenters import Segmenter
from .contour_mask import Contour


def find_segmentation(
    data, dataset, phantom_dataset, endo_target, epi_target, endo_initial, epi_initial
):
    model = "AC"
    model_params_endo = dict(alpha=0.01, beta=10, gamma=0.002)
    model_params_epi = dict(alpha=0.01, beta=10, gamma=0.002)

    ffilter = "gaussian"
    filter_params_endo = {}
    filter_params_epi = {}

    propagator = "initial"
    propagator_params_endo = {}
    propagator_params_epi = {}

    segmenter = Segmenter.setup(model=model, ffilter=ffilter, propagator=propagator)

    all_data = get_data_to_segment(data, dataset, phantom_dataset)

    shape = all_data[endo_target][0].shape
    data.segments[dataset]["endocardium"] = segmenter(
        all_data[endo_target],
        Contour(endo_initial.T, shape=shape),
        model_params=model_params_endo,
        filter_params=filter_params_endo,
        propagator_params=propagator_params_endo,
    )
    data.segments[dataset]["epicardium"] = segmenter(
        all_data[epi_target],
        Contour(epi_initial.T, shape=shape),
        model_params=model_params_epi,
        filter_params=filter_params_epi,
        propagator_params=propagator_params_epi,
    )

    return data


def get_data_to_segment(data, dataset, phantom_dataset):
    """Gets the data that will be segmented and removes the phantom, if needed"""
    magz = data.get_images(dataset, "MagZ")
    magx = data.get_images(dataset, "MagX")
    magy = data.get_images(dataset, "MagY")
    mag = magx + magy + magz
    vel = data.get_images(dataset, "PhaseZ")

    if phantom_dataset != "":
        magz = data.get_bg_images(phantom_dataset, "MagZ")
        magx = data.get_bg_images(phantom_dataset, "MagX")
        magy = data.get_bg_images(phantom_dataset, "MagY")
        mag = mag - (magx + magy + magz)
        vel = vel - data.get_bg_images(phantom_dataset, "PhaseZ")

    return {"mag": mag, "vel": vel}

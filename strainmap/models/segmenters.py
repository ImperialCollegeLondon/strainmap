from typing import Callable, Text, Optional, Union, List
import numpy as np

from skimage.segmentation import (
    active_contour,
    morphological_geodesic_active_contour,
    morphological_chan_vese,
)

from .filters import REGISTERED_FILTERS
from .contour_mask import Contour
from .propagators import PROPAGATORS


SEGMENTERS: dict = {}
""" Dictionary with all the segmenters available in StrainMap."""


def register_segmenter(
    segmenter: Optional[Callable] = None, name: Optional[Text] = None
) -> Callable:
    """ Register a segmenter, so it is available across StrainMap. """

    if segmenter is None:
        return lambda x: register_segmenter(x, name=name)

    name = name if name else segmenter.__name__

    SEGMENTERS[name] = segmenter

    return segmenter


@register_segmenter(name="AC")
def active_contour_model(img: np.ndarray, initial: Contour, params: dict) -> Contour:
    """Segmentation using the active contour model."""
    if img.ndim > 2:
        msg = f"The active contour segmentation model can't perform 3D segmentations."
        raise NotImplementedError(msg)
    else:
        snake = active_contour(img, initial.xy, **params)

    return Contour(snake)


@register_segmenter(name="MorphGAC")
def morphological_geodesic_active_contour_model(
    img: np.ndarray, initial: Contour, params: dict
) -> Union[Contour, List[Contour]]:
    """Segmentation using the morphological geodesic active contour model."""
    iterations = params.pop("iterations", 1000)

    if img.ndim > 2:
        ini = np.array([initial.image] * img.shape[0])
    else:
        ini = initial.image

    snake = morphological_geodesic_active_contour(
        img, iterations=iterations, init_level_set=ini, **params
    )

    if img.ndim > 2:
        result = [Contour(snake[i]) for i in range(img.shape[0])]
    else:
        result = Contour(snake)

    return result


@register_segmenter(name="MorphCV")
def morphological_chan_vese_model(
    img: np.ndarray, initial: Contour, params: dict
) -> Union[Contour, List[Contour]]:
    """Segmentation using the morphological Chan Vese model."""
    iterations = params.pop("iterations", 1000)

    if img.ndim > 2:
        ini = np.array([initial.image] * img.shape[0])
    else:
        ini = initial.image

    snake = morphological_chan_vese(
        img, iterations=iterations, init_level_set=ini, **params
    )

    if img.ndim > 2:
        result = [Contour(snake[i]) for i in range(img.shape[0])]
    else:
        result = Contour(snake)

    return result


class Segmenter(object):
    """Class to create a segmenter that includes a model, a filter and a propagation."""

    def __init__(
        self,
        model: Text = "AC",
        ffilter: Text = "gaussian",
        propagator: Text = "initial",
    ):
        self._model = get_segmenter_model(model)
        self._filter = get_filter(ffilter)
        self._propagator = get_propagator(propagator)
        self._segmenter = (
            self._global_segmenter
            if not self._propagator
            else self._propagated_segmenter
        )

    def __call__(
        self,
        image: np.ndarray,
        initial: Contour,
        mparams: Optional[dict] = None,
        fparams: Optional[dict] = None,
        pparams: Optional[dict] = None,
    ) -> Union[Contour, List[Contour]]:
        return self._segmenter(image, initial, mparams, fparams, pparams)

    def _global_segmenter(
        self,
        image: np.ndarray,
        initial: Contour,
        mparams: Optional[dict] = None,
        fparams: Optional[dict] = None,
        **kwargs,
    ) -> Union[Contour, List[Contour]]:
        """ Segments a single image or array of images at once (3d segmentation)."""

        fimg = self._filter(image, **fparams)
        snake = self._model(fimg, initial, **mparams)

        return snake

    def _propagated_segmenter(
        self,
        image: np.ndarray,
        initial: Contour,
        mparams: Optional[dict] = None,
        fparams: Optional[dict] = None,
        pparams: Optional[dict] = None,
        **kwargs,
    ) -> Union[Contour, List[Contour]]:
        """Segments an array of images propagating the snake from one to the next."""

        fimg = self._filter(image, **fparams)
        if fimg.ndim == 2:
            fimg = np.array([fimg])

        snakes = []
        next_init = initial.to_contour()
        for image in fimg:
            snake = self._model(image, next_init, **mparams)
            snakes.append(snake)
            next_init = self._propagator(initial, snake, **pparams)

        return snakes


def get_segmenter_model(name: Text) -> Callable:
    """Returns the callable of the chosen segmenter model."""
    try:
        return SEGMENTERS[name]
    except KeyError:
        raise KeyError(f"Segmenter {name} does not exists!")


def get_filter(name: Text) -> Callable:
    """Returns the callable of the chosen filter model."""
    try:
        return REGISTERED_FILTERS[name]
    except KeyError:
        raise KeyError(f"Filter {name} does not exists!")


def get_propagator(name: Text) -> Callable:
    """Returns the callable of the chosen propagator model."""
    return PROPAGATORS.get(name, None)

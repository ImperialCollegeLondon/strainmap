from typing import Callable, Text, Optional, Union, List, Dict
import numpy as np

from skimage.segmentation import (
    active_contour,
    morphological_geodesic_active_contour,
    morphological_chan_vese,
)

from .filters import get_filter
from .contour_mask import Contour
from .propagators import get_propagator


SEGMENTERS: dict = {}
""" Dictionary with all the segmenters available in StrainMap."""


def get_segmenter_model(name: Text) -> Callable:
    """Returns the callable of the chosen segmenter model."""
    try:
        return SEGMENTERS[name]
    except KeyError:
        raise KeyError(f"Segmenter {name} does not exists!")


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
def active_contour_model(img: np.ndarray, initial: Contour, params: Dict) -> Contour:
    """Segmentation using the active contour model."""
    if img.ndim > 2:
        msg = f"The active contour segmentation model can't perform 3D segmentations."
        raise NotImplementedError(msg)
    else:
        snake = active_contour(img, initial.xy, **params)

    return Contour(snake)


@register_segmenter(name="MorphGAC")
def morphological_geodesic_active_contour_model(
    img: np.ndarray, initial: Contour, params: Dict
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
        return [Contour(snake[i]) for i in range(img.shape[0])]
    else:
        return Contour(snake)


@register_segmenter(name="MorphCV")
def morphological_chan_vese_model(
    img: np.ndarray, initial: Contour, params: Dict
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
        return [Contour(snake[i]) for i in range(img.shape[0])]
    else:
        return Contour(snake)


class Segmenter(object):
    """Class to create a segmenter that includes a model, a filter and a propagation."""

    @classmethod
    def setup(
        cls,
        model: Text = "AC",
        ffilter: Text = "gaussian",
        propagator: Text = "initial",
    ):
        return cls(
            get_segmenter_model(model), get_filter(ffilter), get_propagator(propagator)
        )

    def __init__(self, model: Callable, ffilter: Callable, propagator: Callable):
        self._model = model
        self._filter = ffilter
        self._propagator = propagator

        if propagator.__name__ == "<lambda>":
            self._segmenter = self._global_segmenter
        else:
            self._segmenter = self._propagated_segmenter

    def __call__(
        self,
        image: np.ndarray,
        initial: Contour,
        model_params: Optional[Dict] = None,
        filter_params: Optional[Dict] = None,
        propagator_params: Optional[Dict] = None,
    ) -> Union[Contour, List[Contour]]:

        mparams = model_params if model_params is not None else {}
        fparams = filter_params if filter_params is not None else {}
        pparams = propagator_params if propagator_params is not None else {}

        return self._segmenter(image, initial, mparams, fparams, pparams)

    def _global_segmenter(
        self,
        image: np.ndarray,
        initial: Contour,
        mparams: Dict,
        fparams: Dict,
        pparams: Dict,
    ) -> Union[Contour, List[Contour]]:
        """ Segments a single image or array of images at once (3d segmentation)."""

        fimg = self._filter(image, **fparams)
        snake = self._model(fimg, initial, **mparams)

        return snake

    def _propagated_segmenter(
        self,
        image: np.ndarray,
        initial: Contour,
        mparams: Dict,
        fparams: Dict,
        pparams: Dict,
    ) -> Union[Contour, List[Contour]]:
        """Segments an array of images propagating the snake from one to the next."""

        fimg = self._filter(image, **fparams)
        if fimg.ndim == 2:
            fimg = np.array([fimg])

        snakes = []
        next_init = initial.to_contour()
        for i, image in enumerate(fimg):
            snake = self._model(image, next_init, **mparams)
            snakes.append(snake)
            next_init = self._propagator(initial, snake, i, **pparams)

        if len(snakes) == 1:
            snakes = snakes[0]

        return snakes

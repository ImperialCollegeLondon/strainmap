from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Text, Union

import numpy as np

from .contour_mask import Contour
from .filters import REGISTERED_FILTERS
from .propagators import REGISTERED_PROPAGATORS

SEGMENTERS: Dict[Text, Callable[..., Any]] = {}
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
def active_contour_model(img: np.ndarray, initial: Contour, **params: Dict) -> Contour:
    """Segmentation using the active contour model."""
    from skimage.segmentation import active_contour

    if img.ndim > 2:
        msg = f"The active contour segmentation model can't perform 3D segmentations."
        raise NotImplementedError(msg)
    else:
        snake = active_contour(img, initial.xy, **params)

    return Contour(snake)


@register_segmenter(name="MorphGAC")
def morphological_geodesic_active_contour_model(
    img: np.ndarray, initial: Contour, **params: Dict
) -> Union[Contour, List[Contour]]:
    """Segmentation using the morphological geodesic active contour model."""
    from skimage.segmentation import morphological_geodesic_active_contour

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
    img: np.ndarray, initial: Contour, **params: Dict
) -> Union[Contour, List[Contour]]:
    """Segmentation using the morphological Chan Vese model."""
    from skimage.segmentation import morphological_chan_vese

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
    """Class to create a segmenter that includes a model, a filter and a propagation.

    - The model is the name of the segmentation algorithm that will search for the
    edges of an image starting with an initial contour. Possible values for the model
    are (all implemented in scikit-image):

        - AC: Active contour model
        - MorphGAC: Morphological geodesic active contour
        - MorphCV: Morphological Chan Vese model

    - The filter is applied to the input images to facilitate the search of the contours
    by enhancing the edges or smoothing noisy areas. Possible values are (again, on
    scikit-image):

        - gaussian
        - inverse_gaussian

    - Finally, the propagator controls how a segmentation done with an image can be used
    to facilitate the segmentation of the next one of the series, using it as the
    initial condition for the model. Possible values are (included in propagators.py):

        - initial: Uses the same initial condition for all images
        - previous: Uses the previous segment as the initial condition for the next one
        - weighted: Uses a weighted average between the initial contour and the previous
                    segment.
    """

    @classmethod
    def setup(
        cls,
        model: Text = "AC",
        ffilter: Text = "gaussian",
        propagator: Text = "initial",
    ) -> Segmenter:
        if propagator is not None:
            return cls(  # type: ignore
                SEGMENTERS.get(model),
                REGISTERED_FILTERS.get(ffilter),
                REGISTERED_PROPAGATORS.get(propagator),
            )
        else:
            return cls(  # type: ignore
                SEGMENTERS.get(model), REGISTERED_FILTERS.get(ffilter)
            )

    def __init__(
        self,
        model: Optional[Callable],
        ffilter: Optional[Callable],
        propagator: Optional[Callable] = None,
    ):
        self._model = model
        self._filter = ffilter
        self._propagator = propagator

        if self._propagator is None:
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
        from copy import copy

        fimg = self._filter(image, **fparams)
        if fimg.ndim == 2:
            fimg = np.array([fimg])

        snakes = []
        next_init = copy(initial)
        for i, image in enumerate(fimg):
            snake = self._model(image, next_init, **mparams)
            snakes.append(snake)
            next_init = self._propagator(  # type: ignore
                initial=initial, previous=snake, step=i, **pparams
            )

        if len(snakes) == 1:
            snakes = snakes[0]

        return snakes

from __future__ import annotations

from copy import copy
from functools import partial, reduce
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import xarray as xr

from ..coordinates import Comp
from .contour_mask import Contour, dilate
from .filters import REGISTERED_FILTERS
from .propagators import REGISTERED_PROPAGATORS

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
replace_threshold: int = 31


SEGMENTERS: Dict[Text, Callable[..., Any]] = {}
""" Dictionary with all the segmenters available in StrainMap."""


def register_segmenter(
    segmenter: Optional[Callable] = None, name: Optional[Text] = None
) -> Callable:
    """Register a segmenter, so it is available across StrainMap."""

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
        msg = "The active contour segmentation model can't perform 3D segmentations."
        raise NotImplementedError(msg)
    else:
        snake = active_contour(img, initial.xy[:, ::-1], **params)[:, ::-1]

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
    """Segmenter that includes a model, a filter and a propagation.

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
        return cls(
            SEGMENTERS[model],
            REGISTERED_FILTERS[ffilter],
            propagator=REGISTERED_PROPAGATORS.get(propagator, None),
        )

    def __init__(
        self, model: Callable, ffilter: Callable, propagator: Optional[Callable] = None
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
        """Segments a single image or array of images at once (3d
        segmentation)."""

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
        """Multi-image Segments via straight-forward snake propagation."""
        from copy import copy

        fimg = self._filter(image, **fparams)
        if fimg.ndim == 2:
            fimg = np.array([fimg])

        snakes = []
        next_init = copy(initial)
        for i, image in enumerate(fimg):
            snake = self._model(image, next_init, **mparams)
            snakes.append(snake)
            assert self._propagator is not None
            next_init = self._propagator(
                initial=initial, previous=snake, step=i, **pparams
            )

        if len(snakes) == 1:
            snakes = snakes[0]

        return snakes


def snakes_segmentation(
    images: xr.DataArray, *, initials: xr.DataArray, **kwargs
) -> xr.DataArray:
    """Find the segmentation of one or more images starting at the initials segments.

    Args:
        images (xr.DataArray): The images for the frames to segment.
        initials (xr.DataArray): Initial segments to start the segmentation with.

    Returns:
        DataArray with the new segments for the requested frames. Dimensions are (side,
        frame, coord, point).
    """
    img = images.sel(comp=Comp.MAG.name)
    frame = img.frame

    segments = initials.expand_dims(axis=1, frame=frame).copy()

    # In frame by frame segmentation, we don't segment if already above threshold
    if frame.size == 1 and frame.item() >= replace_threshold:
        return segments

    rules = (
        _create_rules_one_frame(
            frame.item(), initials, (img.sizes["row"], img.sizes["col"])
        )
        if frame.size == 1
        else _create_rules_all_frames(replace_threshold)
    )

    for side in ("endocardium", "epicardium"):
        segments.loc[{"side": side}] = _simple_segmentation(
            img.data,
            initials.sel(side=side).data,
            model=model,
            model_params=model_params[side],
            ffilter=ffilter,
            filter_params=filter_params[side],
            propagator=propagator,
            propagator_params=propagator_params[side],
            rules=rules[side],
        )

    return segments


def _create_rules_one_frame(
    frame: xr.DataArray, replacement: xr.DataArray, shape: Tuple[int, int]
):
    """Create the rules to apply to the segments to ensure their 'quality'.

    - Expand/contract the contour
    - Replace for the previous segmentation is area change above tolerance.

    Args:
        frame (int): Frame at which to apply the rules.
        replacement (xr.DataArray): Array with all the contours
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
                replacement=Contour(replacement.sel(side=side).data.T, shape=shape),
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

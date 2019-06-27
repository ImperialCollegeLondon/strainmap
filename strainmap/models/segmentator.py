from typing import Callable, Text, Optional, Tuple, Union, List, NoReturn
from abc import ABC, abstractmethod
import numpy as np
from copy import copy

from skimage.segmentation import (
    active_contour,
    morphological_geodesic_active_contour,
    morphological_chan_vese,
)


from .filters import REGISTERED_FILTERS
from .contour_mask import Contour


class SegmentatorBase(ABC):
    def __init__(
        self,
        method: Callable,
        params: Optional[dict] = None,
        sfilter: Text = "gaussian",
        filter_params: Optional[dict] = None,
    ):
        self.name = method.__name__
        self.method = method
        self.params = params if params else {}
        self.filter = REGISTERED_FILTERS[sfilter]
        self.fparams = filter_params if filter_params else {}

    def segment_one_image(
        self,
        img: np.ndarray,
        initial: Contour,
        params: Optional[dict] = None,
        filter_params: Optional[dict] = None,
    ) -> Tuple[Contour, dict, dict]:
        """ Segments a single image. """

        mparams = update_dict(self.params, params)
        fparams = update_dict(self.fparams, filter_params)
        fimg = self.filter(img, **fparams)
        snake = self._run(fimg, initial, mparams)

        return snake, mparams, fparams

    def segment_series(
        self,
        img: List[np.ndarray],
        initial: Contour,
        propagation: Union[Callable, Text] = "none",
        params: Optional[dict] = None,
        filter_params: Optional[dict] = None,
    ) -> Tuple[List[Contour], dict, dict]:
        """ Segments a series of images.

        A propagation method is used to calculate the initial contour for the next
        image in the series. Specific parameters for the segmentation method and the
        filter can be provided as a dictionary where the key is the number of the
        image those parameters apply. """

        if propagation == "3d":
            mparams = update_dict(self.params, params)
            fparams = update_dict(self.fparams, filter_params)

            fimg = np.array([self.filter(image, **fparams) for image in img])
            snakes = self._run3d(fimg, initial, mparams)

        else:
            params = params if params else {}
            filter_params = filter_params if filter_params else {}
            snakes = []
            mparams = {}
            fparams = {}

            for i, image in enumerate(img):

                snake, mparams[i], fparams[i] = self.segment_one_image(
                    image, initial, params.get(i, None), filter_params.get(i, None)
                )
                snakes.append(snake)

        return snakes, mparams, fparams

    def segment_combined(
        self,
        img: Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]],
        initial: Tuple[Contour, Contour],
        propagation: Union[Callable, Text] = "none",
        params: Union[dict, Tuple[dict, dict], None] = None,
        filter_params: Union[dict, Tuple[dict, dict], None] = None,
    ) -> Tuple[
        Tuple[List[Contour], List[Contour]], Tuple[dict, dict], Tuple[dict, dict]
    ]:
        """ Segments simultaneously the epi- and endocardium at each timestep.

        This function performs the segmentation for the epi- and the endocardium at
        each timestep before moving to the next one. Two initial conditions are needed
        and, optionally, two sets of input images.

        This method allows for more elaborate propagation methods, for example
        considering the velocity to estimate the next initial condition.
        """
        if isinstance(img, List):
            img = (img, img)

        if isinstance(params, dict):
            params = (params, params)
        elif not params:
            params = ({}, {})

        if isinstance(filter_params, dict):
            filter_params = (filter_params, filter_params)
        elif not filter_params:
            filter_params = ({}, {})

        snakes: Tuple[List[Contour], List[Contour]] = ([], [])
        mparams: Tuple[dict, dict] = ({}, {})
        fparams: Tuple[dict, dict] = ({}, {})

        new_init = copy(initial)
        for i in range(len(img[0])):
            snake, mparams[0][i], fparams[0][i] = self.segment_one_image(
                img[0][i],
                new_init[0],
                params[0].get(i, None),
                filter_params[0].get(i, None),
            )
            snakes[0].append(snake)

            snake, mparams[1][i], fparams[1][i] = self.segment_one_image(
                img[1][i],
                new_init[1],
                params[1].get(i, None),
                filter_params[1].get(i, None),
            )
            snakes[1].append(snake)

            new_init = (snakes[0][-1], snakes[1][-1])

        return snakes, mparams, fparams

    @abstractmethod
    def _run(self, img: np.ndarray, initial: Contour, params: dict) -> Contour:
        """ Method that calls the segmentation function.

        This function has to be adapted by each segmentator to ensure that the Contour
        input is used in its correct form and that the output snake is returned as a
        Contour or list of contours in the case of 3D segmentations. """

    @abstractmethod
    def _run3d(self, img: np.ndarray, initial: Contour, params: dict) -> List[Contour]:
        """ Method that calls the segmentation function for 3D segmentations.

        This function has to be adapted by each segmentator to ensure that the Contour
        input is used in its correct form and that the output snake is returned as a
        list of contours. """


class ActiveContour(SegmentatorBase):
    def __init__(
        self,
        params: Optional[dict] = None,
        sfilter: Text = "gaussian",
        filter_params: Optional[dict] = None,
    ):
        super().__init__(
            method=active_contour,
            params=params,
            sfilter=sfilter,
            filter_params=filter_params,
        )

    def _run(self, img: np.ndarray, initial: Contour, params: dict) -> Contour:
        snake = self.method(img, initial.xy, **params)
        return Contour(snake)

    def _run3d(self, img: np.ndarray, initial: Contour, params: dict) -> NoReturn:
        msg = f"The '{self.name}' segmentation method can't perform 3D segmentations."
        raise NotImplementedError(msg)


class MorphologicalGeodesicActiveContour(SegmentatorBase):
    def __init__(
        self,
        params: Optional[dict] = None,
        sfilter: Text = "inverse_gaussian",
        filter_params: Optional[dict] = None,
    ):
        super().__init__(
            method=morphological_geodesic_active_contour,
            params=params,
            sfilter=sfilter,
            filter_params=filter_params,
        )

    def _run(self, img: np.ndarray, initial: Contour, params: dict) -> Contour:
        iterations = params.pop("iterations", 1000)
        snake = self.method(
            img, iterations=iterations, init_level_set=initial.xy2d, **params
        )
        return Contour(snake)

    def _run3d(self, img: np.ndarray, initial: Contour, params: dict) -> List[Contour]:
        ini = np.array([initial.xy2d] * len(img.shape[0]))
        iterations = params.pop("iterations", 1000)
        snake = self.method(img, iterations=iterations, init_level_set=ini, **params)
        return [Contour(snake[i]) for i in range(len(img.shape[0]))]


class MorphologicalChanVese(SegmentatorBase):
    def __init__(
        self,
        params: Optional[dict] = None,
        sfilter: Text = "inverse_gaussian",
        filter_params: Optional[dict] = None,
    ):
        super().__init__(
            method=morphological_chan_vese,
            params=params,
            sfilter=sfilter,
            filter_params=filter_params,
        )

    def _run(self, img: np.ndarray, initial: Contour, params: dict) -> Contour:
        iterations = params.pop("iterations", 1000)
        snake = self.method(
            img, iterations=iterations, init_level_set=initial.xy2d, **params
        )
        return Contour(snake)

    def _run3d(self, img: np.ndarray, initial: Contour, params: dict) -> List[Contour]:
        ini = np.array([initial.xy2d] * len(img.shape[0]))
        iterations = params.pop("iterations", 1000)
        snake = self.method(img, iterations=iterations, init_level_set=ini, **params)
        return [Contour(snake[i]) for i in range(len(img.shape[0]))]


def update_dict(original: dict, new: Optional[dict] = None) -> dict:
    """ Updates only existing keys in the original dict with the values of the new. """
    return {k: new.get(k, original.get(k)) for k in original} if new else copy(original)

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import decouple
import numpy as np
import xarray as xr
from pubsub import pub
from skimage import measure
from tensorflow import keras


def ai_segmentation(
    images: xr.DataArray, *, initials: xr.DataArray, points: int, **kwargs
) -> xr.DataArray:
    """Performs the segmentation using the AI segmenter.

    Images are first normalized, then are fed into the AI segmenter - which is loaded
    from disk if it is the first time is used -, and finally the output (2D masks) is
    processed to get the contours. If contours cannot be found for a given frame, np.nan
    is used.

    Args:
        images: Input DataArray with the images to segment. Channels must be arranged
            as (Mag, phase X, phase Y, phase Z).
        points: Number of points for each contour.

    Returns:
        DataArray with the new segments for the requested frames. Dimensions are (side,
        frame, coord, point).
    """
    crop = 128
    max_rows = images.sizes["row"] - crop
    max_cols = images.sizes["col"] - crop

    normalized = Normal.run(
        images.sel(row=range(crop, max_rows), col=range(crop, max_cols))
        .transpose("frame", "col", "row", "comp")
        .data,
        method="ubytes",
    )
    labels = UNet.factory().predict(normalized)
    segments = labels_to_contours(labels, points, initials.data - crop) + crop
    return xr.DataArray(
        segments,
        dims=["side", "frame", "coord", "point"],
        coords={
            "side": ["endocardium", "epicardium"],
            "frame": images.frame,
            "coord": ["col", "row"],
        },
    )


class UNet:
    _unet: Optional[UNet] = None

    @classmethod
    def reset_ai(cls) -> None:
        """Removes the cached AI, so a new call to factory will re-loaded from disk"""
        cls._unet = None

    @classmethod
    def factory(cls, model_location: Optional[Path] = None) -> UNet:
        """Factory method to load the model from a folder.

        This class is singleton, so if the model has already been loaded, the existing
        one is returned.

        Args:
            model_location: Path to the location of a keras model. If None, the
            environmental variable STRAINMAP_AI_MODEL is checked.

        Raises:
            RuntimeError if neither the model_location nor the STRAINMAP_AI_MODEL
            point to a valid location.

        Returns:
            The loaded keras model
        """
        if cls._unet is not None:
            return cls._unet

        path = (
            model_location
            if model_location is not None
            else decouple.config("STRAINMAP_AI_MODEL", default=None)
        )
        if not path:
            pub.sendMessage("segmentation.select_ai")
            path = decouple.config("STRAINMAP_AI_MODEL", default=None)

        if not path:
            raise RuntimeError("No path provided for the AI model.")

        logger = logging.getLogger(__name__)
        logger.info("Loading AI model. Please, wait...")
        model = keras.models.load_model(path)
        logger.info("Done!")
        return cls(model=model)

    def __new__(cls, model: keras.models.Model) -> UNet:
        if cls._unet is None:
            cls._unet = super(UNet, cls).__new__(cls)
            cls._unet.model = model
            pub.subscribe(cls.reset_ai, "segmentation.reset_ai")

        return cls._unet

    def __init__(self, *args, **kwargs):
        self.model: keras.models.Model

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Use the model to predict the labels given the input images.

        Args:
            images: Array of images we want to infer the labels from. Must have shape
            (n, h, w, c) and be normalised. n is the number or images, (h, w) are the
            height and width respectively and c the number of channels per image.
            If h, w and c do not match those the model has been trained for, the
            calculation will fail.

        Returns:
            Integer array of shape (n, h, w) with the predicted labels, 1 for the mask
            area and 0 for the rest.
        """
        result = self.model.predict(
            x=images,
            batch_size=8,
            verbose=1,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return (result[..., 0] > 0.5).astype(np.int8)


"""
Pre-processing and data augmentation routines
"""


class Normal:
    _normalizers: Dict[str, Callable] = {"None": lambda x: x}

    @classmethod
    def avail(cls) -> Tuple[str, ...]:
        """Available normalizers"""
        return tuple(cls._normalizers.keys())

    @classmethod
    def register(cls, f: Callable) -> Callable:
        """Add input normalizer to the registry"""
        cls._normalizers[f.__name__] = f
        return f

    @classmethod
    def run(cls, data: np.ndarray, method: str = "None") -> np.ndarray:
        """Run the normalization of the data using the chosen normalizer.

        Args:
            data: Array with the data to normalize.
            method: Normalizer to use.

        Raises:
            Key Error: If the chosen normalizer does not exist in the registry.

        Return:
            A new array with the same shape than input and the data normalized.
        """
        return cls._normalizers[method](data)


@Normal.register
def zero2one(data: np.ndarray) -> np.ndarray:
    """Data is normalized to have minimum equal to zero and maximum equal to one.

    Args:
        data: Array with the data to normalize.

    Return:
        A new array with the same shape than input and the data normalized.
    """
    amax = np.amax(data)
    amin = np.amin(data)
    return (data - amin) / (amax - amin)


@Normal.register
def ubytes(data: np.ndarray) -> np.ndarray:
    """Data is normalized to unsigned byte size (0, 255).

    Args:
        data: Array with the data to normalize.

    Return:
        A new array with the same shape than input and the data normalized.
    """
    return (data / data.max() * np.iinfo(np.uint8).max).round().astype(np.uint8)


@Normal.register
def zeromean_unitvar(data: np.ndarray) -> np.ndarray:
    """Data is normalized to have mean equal to zero and variance equal to one.

    Args:
        data: Array with the data to normalize.

    Return:
        A new array with the same shape than input and the data normalized.
    """
    return (data - np.mean(data)) / np.std(data)


"""
Post-processing routines
"""


def crop_roi(labels: np.ndarray, margin: int = 70) -> np.ndarray:
    """Crop the inferred labels to cover only the region of the bigger blob.

    The goal here is to remove spurious contours found by the AI away form the
    ventricle and that could make a proper analysis harder.

    The process is done in several steps:
        1- Image is transformed to binary
        2- Contours are found
        3- The contour enclosing the biggest area is identified and the centroid found
        4- A region around that centroid is cropped.
        5- An array of labels with the same shape as input but with only the labels
            within that cropped region is returned

    Args:
        labels: Array of inferred labels of shape (h, w).
        margin: Pixels around the centroid of the bigger blob in each direction defining
            the square region of interest.

    Raises:
        ValueError: If no contours could be found.

    Returns:
        Array of labels cropped to the region of the biggest blob.
    """
    arr = labels.astype(np.uint8)
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY)

    if np.count_nonzero(binary) == 0:
        raise ValueError("No masks to find contour found available.")

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest = sorted(contours, key=cv2.contourArea)[-1]
    M = cv2.moments(largest)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    roi = cv2.rectangle(
        np.zeros_like(labels, dtype=np.uint8),
        (cX - margin, cY - margin),
        (cX + margin, cY + margin),
        1,
        -1,
    )
    return arr * roi


def add_ellipse(labels: np.ndarray) -> np.ndarray:
    """Draws an ellipse in the labels closing any gap in the myocardium mask.

    The myocardium identified by the AI might be incomplete,leaving gaps in the
    mask. This function fits the pixels identified as myocardium with a thin
    ellipse.

    If the myocardium has been correctly identified and it is vaguely circular, this
    function makes no difference as the drawn ellipse will fit into the the existing
    mask. If there are gaps, they will be filled with the ellipse data, closing the
    mask.

    The only caveat is that, if the myocardium has been correctly identified but it is
    not roughly circular such that an ellipse wholy fit within, this function will
    artificially add pixel points outside the mask that should not be there.

    Args:
        labels: Array of inferred labels of shape (h, w).

    Raises:
        RuntimeError: If an ellipse cannot be traced.

    Returns:
        An array with the same shape and the ellipse added to it.
    """
    coords = np.asarray(np.nonzero(labels)).T
    ellipse = measure.EllipseModel()

    if not ellipse.estimate(coords):
        raise RuntimeError("An ellipse could not be drawn.")

    # For drawing the ellipse, we need the parameters as integers
    xc, yc, a, b, theta = ellipse.params
    xc = int(round(xc))
    yc = int(round(yc))
    a = int(round(a))
    b = int(round(b))
    theta = 90 - np.rad2deg(theta)

    return cv2.ellipse(
        img=labels.copy().astype(np.int8),
        center=(yc, xc),
        axes=(a, b),
        angle=theta,
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=3,
    )


def get_contours(labels: np.ndarray) -> List[np.ndarray]:
    """Extract the contours out of the labels mask.

    The two contours enclosing the largest area are assumed to be those corresponding to
    the epicardium and the endocardium, respectively.

    Args:
        labels: Array of inferred labels of shape (h, w).

    Raises:
        ValueError: If less than two contours are found.

    Returns:
        List of two arrays corresponding to the epicardium and de endocardium contours.
        Each contour has shape (2, p), with p the number of points per contour and,
        in general, different for epi and endocarcium. All points of the contours are
        part of the input labels (i.e., correspond to points where the labels array has
        a value of 1).
    """
    arr = labels.astype(np.uint8)
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    s_contours = sorted(contours, key=cv2.contourArea)[-2:]

    if len(s_contours) < 2:
        raise ValueError("Not enough contours found.")

    return [np.squeeze(s).T[::-1] for s in s_contours]


def interpolate_contour(contour: np.ndarray, points: int) -> np.ndarray:
    """Interpolates a closed contour to the chosen number of points.

    Args:
        contour: The contour to interpolate of shape (2, p)
        points: The number of required points in the final contour.

    Returns:
        Array with the interpolated contour.
    """
    xp = np.linspace(0, 1, contour.shape[1], endpoint=False)
    x = np.linspace(0, 1, points)
    y1 = np.interp(x, xp, contour[0], period=1)
    y2 = np.interp(x, xp, contour[1], period=1)
    return np.array([y1, y2])


def labels_to_contours(
    labels: np.ndarray, points: int = 360, default: Optional[np.ndarray] = None
) -> np.ndarray:
    """Process the labels produced by the AI and extract the epi and end contours.

    For those inferred labels for which it is not possible to extract two contours,
    the ones of the previous frame are copied, if they exist, or contours filled with
    NaN are produced otherwise. A warning is raised in either case.

    Args:
        labels: Array of inferred labels of shape (n, h, w).
        points: Number of points for each contour.
        default: Default segments if none are found.

    Returns:
        Array of contours found out of the corresponding labels, of shape (2, n, 2, p),
        meaning, respectively: side (epi- and endocardium), number of images (frames)
        coordinates of each contour and the points.
    """
    contours = []
    not_found = 0
    default = default if default is not None else np.full((2, 2, points), np.nan)
    for arr in labels:
        try:
            sanitized = add_ellipse(crop_roi(arr))
            contours.append(
                np.array(
                    [interpolate_contour(c, points) for c in get_contours(sanitized)]
                )
            )
        except (RuntimeError, ValueError):
            contours.append(contours[-1].copy() if len(contours) > 0 else default)
            not_found += 1

    if not_found > 0:
        warnings.warn(f"Contours not found for {not_found} images.", RuntimeWarning)

    return np.array(contours).transpose((1, 0, 2, 3))

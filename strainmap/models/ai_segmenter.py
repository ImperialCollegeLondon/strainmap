from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Tuple, List
from pathlib import Path
from functools import partial, reduce
import tempfile
import warnings

import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorlayer import prepro
import toml
import cv2
from skimage import measure


@dataclass
class UNet:
    """This class defines the architecture of the UNet model.

    It contains the specific definition of the layers of the neural network as well as
    general parameters needed to define those layers. It also exposes an interface for
    compiling the model, running the training and the inference.
    """

    model_name: str = "Unet"
    img_height: int = 512
    img_width: int = 512
    nclasses: int = 1
    filters: int = 64
    batch_size: int = 8
    epochs: int = 5
    verbose: int = 1
    callbacks: Optional[list] = None
    validation_split: float = 0.05
    shuffle: bool = True
    initial_epoch: int = 0
    steps_per_epoch: Optional[int] = None
    imgchannel: int = 4
    model_file: Optional[str] = None
    _model: Optional[Model] = field(default=None, init=False)

    @classmethod
    def factory(cls, config_file: Optional[str] = None) -> UNet:
        """Factory method to create th net out of a config file.

        Args:
            config_file: Path to a toml-formated config file containing the
                configuration details to create the network.
        """
        path = (
            Path(config_file)
            if config_file is not None
            else Path(__file__).parent / "ai_config.toml"
        )
        config = toml.load(path)["Net"]
        config = {k: v if v != "None" else None for k, v in config.items()}
        return cls(**config)

    @property
    def model(self) -> Model:
        """Keras Model object."""
        if self._model is None:
            raise RuntimeError("Model has not been created, yet.")
        else:
            return self._model

    def _conv_block(
        self,
        tensor,
        nfilters,
        size=3,
        padding="same",
        initializer="he_normal",
        repetitions=2,
    ):
        """Create a convolution block.

        Each block consist on a number of repetitions of 2D convolution, batch
        normalization and activation.
        """
        x = tensor
        for i in range(repetitions):
            x = layers.Conv2D(
                filters=nfilters,
                kernel_size=(size, size),
                padding=padding,
                kernel_initializer=initializer,
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

        return x

    def _deconv_block(
        self, tensor, residual, nfilters, size=3, padding="same", strides=(2, 2)
    ):
        """Create a deconvolution block.

        The block consist on a transpose 2D convolution followed by a concatenation
        with the residual and a convolution block.
        """
        y = layers.Conv2DTranspose(
            nfilters, kernel_size=(size, size), strides=strides, padding=padding
        )(tensor)
        y = layers.concatenate([y, residual], axis=3)
        y = self._conv_block(y, nfilters)
        return y

    def _modelstruct(self):
        """Creates the UNet model out of a sequence of layers."""
        h = self.img_height
        w = self.img_width
        nclasses = self.nclasses
        filters = self.filters

        # Input layer
        input_layer = layers.Input(shape=(h, w, self.imgchannel), name="image_input")

        # Down
        conv1 = self._conv_block(input_layer, nfilters=filters)
        conv1_out = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self._conv_block(conv1_out, nfilters=filters * 2)
        conv2_out = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self._conv_block(conv2_out, nfilters=filters * 4)
        conv3_out = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self._conv_block(conv3_out, nfilters=filters * 8)
        conv4_out = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        conv4_out = layers.Dropout(0.5)(conv4_out)
        conv5 = self._conv_block(conv4_out, nfilters=filters * 16)
        conv5 = layers.Dropout(0.5)(conv5)

        # Up
        deconv6 = self._deconv_block(conv5, residual=conv4, nfilters=filters * 8)
        deconv6 = layers.Dropout(0.5)(deconv6)
        deconv7 = self._deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
        deconv7 = layers.Dropout(0.5)(deconv7)
        deconv8 = self._deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
        deconv9 = self._deconv_block(deconv8, residual=conv1, nfilters=filters)

        # Output layer
        output_layer = layers.Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
        output_layer = layers.BatchNormalization()(output_layer)
        output_layer = layers.Activation("sigmoid")(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.model_name)
        return model

    def compile_model(self, print_summary=True, model_file: Optional[str] = "default"):
        """Creates and compiles the Model object.

        Args:
            print_summary: If a summary for the model should be printed.
            model_file: If provided, this should be either the path of a h5 file
                containing the weights for the model or "default" to use the model
                indicated in the config file.

        Returns:
            None
        """
        self._model = self._modelstruct()
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["acc"]
        )

        if print_summary:
            self.model.summary()

        if model_file == "default" and self.model_file is not None:
            self.model.load_weights(self.model_file)
        elif model_file is not None:
            self.model.load_weights(model_file)

    def train(
        self, images: np.ndarray, labels: np.ndarray, model_file: Optional[Path] = None
    ) -> None:
        """Train a model to best fit the labels based on the input images.

        Args:
            images: Array of images that serve as input to the model of shape
                (n, h, w, c)
            labels: Array of labels that represent the expected output of the model of
                shape (N, h, w).
            model_file: If provided, if should be the path to the h5 file where the
                weights will be saved once the training is complete.

        Returns:
            None
        """
        self.model.fit(
            x=images,
            y=labels[..., None],
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_split=self.validation_split,
            shuffle=self.shuffle,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
        )

        if model_file is not None:
            self.model.save_weights(model_file)

    def infer(self, images: np.ndarray) -> np.ndarray:
        """Use the model to predict the labels given the input images.

        Args:
            images: Array of images we want to infer the labels from.

        Returns:
            Array with the predicted labels.
        """
        result = self.model.predict(
            x=images,
            batch_size=self.batch_size,
            verbose=self.verbose,
            steps=None,
            callbacks=self.callbacks,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return 1.0 * (result[..., 0] > 0.5)


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
def zeromean_unitvar(data: np.ndarray) -> np.ndarray:
    """Data is normalized to have mean equal to zero and variance equal to one.

    Args:
        data: Array with the data to normalize.

    Return:
        A new array with the same shape than input and the data normalized.
    """
    return (data - np.mean(data)) / np.std(data)


class DataAugmentation:
    _method: Dict[str, Callable] = {
        "vertical_flip": partial(prepro.flip_axis_multi, axis=1),
        "horizontal_flip": partial(prepro.flip_axis_multi, axis=2),
        "rotation": prepro.rotation_multi,
        "elastic": prepro.elastic_transform_multi,
        "shift": prepro.shift_multi,
        "shear": prepro.shear_multi,
        "zoom": prepro.zoom_multi,
    }

    @classmethod
    def factory(cls, config_file: Optional[str] = None) -> DataAugmentation:
        """Factory method to create objects based on the steps of a config file.

        Args:
            config_file: Path to a toml-formated config file containing the steps for
                the data augmentation.
        """
        path = (
            Path(config_file)
            if config_file is not None
            else Path(__file__).parent / "ai_config.toml"
        )
        config = toml.load(path)["augmentation"]
        steps = {c.pop("method"): c for c in config["active"]}
        return cls(steps, config["times"], config["axis"], config["include_original"])

    def __init__(
        self, steps: Dict[str, Dict], times: int, axis: int, include_original: bool
    ):
        """Create a data augmentation object.

        Args:
            steps: Dictionary of transformation functions to apply and their input
                parameters.
            times: The number of times the data should be transformed.
            axis: The transformed data will be concatenated along this axis.
            include_original: If the original data should be part of the augmented set.
        """
        self.steps = [partial(self._method[k], **v) for k, v in steps.items()]
        self.times = times
        self.axis = axis
        self.include_original = include_original

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply all the transformation steps to the input array sequentially.

        Args:
            data: Array to transform, of shape (N, h, w), with N the number of images,
                and (h, w) their height and width, respectively.

        Return:
            The transformed array resulting from the cummulative application of all the
            transformation steps.
        """
        return reduce(lambda d, f: f(d), self.steps, data)

    def augment(self, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Augment the original array by applying the transformations several times.

        Each transformation is saved to a temporary file to save memory while the rest
        of transformations are completed. All the files are read and the arrays put
        together at the end of the process.

        Args:
            images: Array with the images, of shape (n, h, w, c), with N the number of
                images, (h, w) their height and width, respectively, and c the number
                of channels per image.
            labels: Array with the labels corresponding to each image, of shape
                (Nn h, w) and the same meaning than the images.

        Return:
            Tuple with two new arrays, the augmented images and the corresponding
            augmented labels. The shape of the arrays will be the same than the input
            ones except along "axis" where they will be "times" bigger or
            "times + 1" if "include_original" is true.
        """
        files = []
        channels = images.shape[-1]
        data = self._group(images, labels)
        with tempfile.TemporaryDirectory() as r:
            root = Path(r)

            # Save the original data, if needed
            if self.include_original:
                files.append(root / "original.npy")
                np.save(files[-1], data)

            # Transform data and save it a number of times
            for i in range(self.times):
                t = self.transform(data)
                files.append(root / f"{i}.npy")
                np.save(files[-1], t)
                del t

            # Read the data and concatenate all the arrays along the chosen axis
            images, labels = zip(*[self._ungroup(np.load(f), channels) for f in files])
            images = [Normal.run(img, method="zeromean_unitvar") for img in images]
            return np.concatenate(images, axis=self.axis), np.concatenate(
                labels, axis=self.axis
            )

    @staticmethod
    def _group(images: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Group images and labels and reshape the result prior to augmentation.

        The channels are unstacked, multiplying the number of images by c, and the
        labels are concatenated at the end, resulting in an array of
        shape (n*(c+1), h, w)

        Args:
            images: Array with the images, of shape (n, h, w, c), with n the number of
                images, (h, w) their height and width, respectively, and c the number
                of channels per image.
            labels: Array with the labels corresponding to each image, of shape
                (n, h, w) and the same meaning than the images.

        Return:
            Array grouping channels and labels in the first dimension with
            shape (n*(c+1), h, w)
        """
        return np.concatenate(
            [images[..., i] for i in range(images.shape[-1])]
            + [
                labels,
            ],
            axis=0,
        )

    @staticmethod
    def _ungroup(data: np.ndarray, channels: int) -> Tuple[np.ndarray, np.ndarray]:
        """Ungroup data into images with several channels and labels.

        The number of individual images (with c channels each) is given by:

            n = m / (c + 1)

        Args:
            data: Array of shape (m, h, w).
            channels: Number of channels per image.

        Return:
            A tuple with two arrays. The first one contains the images, of shape
            (n, h, w, c), with n the number of images, (h, w) their height and width,
            respectively, and c the number of channels per image. The second contains
            the labels corresponding to each image, of shape (n, h, w) and the same
            meaning than the images.
        """
        n = int(data.shape[0] / (channels + 1))
        images = np.stack([data[i * n : (i + 1) * n] for i in range(channels)], axis=-1)
        return images, data[-n:].copy()


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
        They will not have the same number of points, in general.
    """
    arr = labels.astype(np.uint8)
    _, binary = cv2.threshold(arr, 0.5, 1, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    s_contours = sorted(contours, key=cv2.contourArea)[-2:]

    if len(s_contours) < 2:
        raise ValueError("Not enough contours found.")

    return s_contours


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


def labels_to_contours(labels: np.ndarray, points: int = 361) -> np.ndarray:
    """Process the labels produced by the AI and extract the epi and end contours.

    For those inferred labels for which it is not possible to extract two contours,
    contours filled with NaN are produced. A warning is raised in that case.

    Args:
        labels: Array of inferred labels of shape (n, h, w).
        points: Number of points for each contour.

    Returns:
        Array of contours found out of the corresponding labels, of shape (n, 2, 2, p),
        meaning, respectively: number of images, side (epi- and endocardium),
        coordinates of each contour and the points.
    """
    contours = []
    not_found = 0
    for arr in labels:
        try:
            sanitized = add_ellipse(crop_roi(arr))
            contours = np.array(
                [interpolate_contour(c, points) for c in get_contours(sanitized)]
            )
        except (RuntimeError, ValueError):
            contours.append(np.full((2, 2, points), np.nan))
            not_found += 1

    if not_found > 0:
        warnings.warn(f"Contours not found for {not_found} images.", RuntimeWarning)

    return np.array(contours)

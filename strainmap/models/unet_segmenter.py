from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
from tensorflow.python.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Activation,
    Dropout,
    Reshape,
)
from tensorflow.python.keras.models import Model


@dataclass
class UNet:
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
    no_chn_split: bool = True
    lbl_reshape_into1d: bool = True
    lblchnadded: bool = False
    _model: Optional[Model] = field(default=None, init=False)

    @property
    def model(self) -> Model:
        """Keras Model object."""
        if self._model is None:
            raise RuntimeError("Model has not been created, yet.")
        else:
            return self._model

    def conv_block(
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
            x = Conv2D(
                filters=nfilters,
                kernel_size=(size, size),
                padding=padding,
                kernel_initializer=initializer,
            )(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        return x

    def deconv_block(
        self, tensor, residual, nfilters, size=3, padding="same", strides=(2, 2)
    ):
        """Create a deconvolution block.

        The block consist on a transpose 2D convolution followed by a concatenation
        with the residual and a convolution block.
        """
        y = Conv2DTranspose(
            nfilters, kernel_size=(size, size), strides=strides, padding=padding
        )(tensor)
        y = concatenate([y, residual], axis=3)
        y = self.conv_block(y, nfilters)
        return y

    def modelstruct(self):
        """Creates the UNet model out of a sequence of layers."""
        h = self.img_height
        w = self.img_width
        nclasses = self.nclasses
        filters = self.filters

        # Input layer
        input_layer = Input(shape=(h, w, self.imgchannel), name="image_input")

        # Down
        conv1 = self.conv_block(input_layer, nfilters=filters)
        conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self.conv_block(conv1_out, nfilters=filters * 2)
        conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self.conv_block(conv2_out, nfilters=filters * 4)
        conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = self.conv_block(conv3_out, nfilters=filters * 8)
        conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv4_out = Dropout(0.5)(conv4_out)
        conv5 = self.conv_block(conv4_out, nfilters=filters * 16)
        conv5 = Dropout(0.5)(conv5)

        # Up
        deconv6 = self.deconv_block(conv5, residual=conv4, nfilters=filters * 8)
        deconv6 = Dropout(0.5)(deconv6)
        deconv7 = self.deconv_block(deconv6, residual=conv3, nfilters=filters * 4)
        deconv7 = Dropout(0.5)(deconv7)
        deconv8 = self.deconv_block(deconv7, residual=conv2, nfilters=filters * 2)
        deconv9 = self.deconv_block(deconv8, residual=conv1, nfilters=filters)

        # Output layer
        output_layer = Conv2D(filters=nclasses, kernel_size=(1, 1))(deconv9)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Reshape(
            (h * w, nclasses),
            input_shape=(h, w, nclasses),
        )(output_layer)
        output_layer = Activation("sigmoid")(output_layer)

        model = Model(inputs=input_layer, outputs=output_layer, name=self.model_name)
        return model

    def compile_model(self, print_summary=True, weigths: Optional[Path] = None):
        """Creates and compiles the Model object.

        Args:
            print_summary: If a summary for the model should be printed.
            weigths: If provided, this should be a Path object pointing to a h5 file
                containing the weigths for the model.

        Returns:
            None
        """
        self._model = self.modelstruct()
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["acc"]
        )

        if print_summary:
            self.model.summary()

        if weigths is not None:
            self.model.load_weights(weigths)

    def reshapelbl(self, lbl):
        """Reshape the array of labels (the masks).

        TODO: Vectorize calculation
        """
        lbl_reshape = np.zeros(
            [lbl.shape[0], lbl.shape[1] * lbl.shape[2], lbl.shape[3]]
        )
        for a in range(lbl.shape[0]):
            for d in range(lbl.shape[3]):
                lbl_reshape[a, :, d] = np.reshape(
                    lbl[a, :, :, d], (lbl.shape[1] * lbl.shape[2])
                )
        return lbl_reshape

    def revertReshapeLbl(self, lbl):
        """Revert the re-shaping, returning the labels to the original shape.

        TODO: Vectorize calculation
        """
        imgsize2d = (self.img_height, self.img_width)
        lbl_revert = np.zeros([lbl.shape[0], imgsize2d[0], imgsize2d[1], lbl.shape[2]])
        for a in range(lbl.shape[0]):
            for d in range(lbl.shape[2]):
                lbl_revert[a, :, :, d] = np.reshape(
                    lbl[a, :, d], (imgsize2d[0], imgsize2d[1])
                )
        if self.no_chn_split:
            return lbl_revert[:, :, :, 0]
        else:
            return lbl_revert

    def train(
        self, images: np.ndarray, labels: np.ndarray, filename: Optional[Path] = None
    ) -> None:
        """Train a model to best fit the labels based on the input images.

        Args:
            images: Array of images that serve as input to the model.
            labels: Array of labels that represent the expected output of the model.
            filename: If provided, if should be a Path object where the weigths will be
                saved once the training is complete.

        Returns:
            None
        """
        self.model.fit(
            x=images,
            y=self.reshapelbl(labels),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            callbacks=self.callbacks,
            validation_split=self.validation_split,
            shuffle=self.shuffle,
            initial_epoch=self.initial_epoch,
            steps_per_epoch=self.steps_per_epoch,
        )

        if filename is not None:
            self.model.save_weights(filename)

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
            verbose=self.epochs,
            steps=None,
            callbacks=self.callbacks,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        result = 1.0 * (result > 0.5)
        return self.revertReshapeLbl(result)


def reorgarr(x, y):
    """
    this assumes that
    x is of shape 50*x*x*4
    y is of shape 50*x*x
    """
    # if not (len(x.shape) == 4 and x.shape[0] == 50 and x.shape[3] == 4):
    #     print(x.shape)
    #     raise ValueError("x shape must be 50*:*:*4")
    # if not (len(y.shape) == 3 and y.shape[0] == 50):
    #     print(y.shape)
    #     raise ValueError("y shape must be 50*:*:")
    inputsize = x.shape[1:3]

    integrated = np.zeros([0, inputsize[0], inputsize[1], 1])
    for i in range(x.shape[0]):
        for layeri in range(x.shape[3]):
            integrated = np.append(
                integrated, (x[i, :, :, layeri])[np.newaxis, :, :, np.newaxis], axis=0
            )
        integrated = np.append(
            integrated, (y[i, :, :])[np.newaxis, :, :, np.newaxis], axis=0
        )
    return integrated

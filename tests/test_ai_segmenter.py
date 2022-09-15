from unittest.mock import MagicMock

import pytest


@pytest.fixture
def data_shape():
    return 64, 32, 4


@pytest.fixture
def keras_model(data_shape):
    import numpy as np
    from tensorflow import keras

    train_input = np.random.random((100, *data_shape))
    train_target = np.random.random((100, data_shape[0], data_shape[1], 1))

    # Inputs are images of unknown size and 4 channels
    inputs = keras.Input(shape=(None, None, data_shape[-1]))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(train_input, train_target)
    return model


def test_add_ellipse():
    import cv2
    import numpy as np

    from strainmap.models.ai_segmenter import add_ellipse

    # Inputs is an incomplete ellipse
    labels = cv2.ellipse(
        img=np.zeros((20, 20), dtype=np.int8),
        center=(9, 9),
        axes=(4, 6),
        angle=30,
        startAngle=0,
        endAngle=270,
        color=1,
        thickness=1,
    )

    # Output should be a close ellipse
    expected_nz = np.nonzero(
        cv2.ellipse(
            img=np.zeros((20, 20), dtype=np.int8),
            center=(9, 9),
            axes=(4, 6),
            angle=30,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=2,
        )
    )
    actual = add_ellipse(labels)

    assert (actual[expected_nz] == 1).all()


def test_get_contours():
    import cv2
    import numpy as np

    from strainmap.models.ai_segmenter import get_contours

    labels = np.zeros((20, 20), dtype=np.int8)
    labels[4:15, 4:15] = 1

    with pytest.raises(ValueError):
        get_contours(labels)

    labels = cv2.ellipse(
        img=np.zeros((20, 20), dtype=np.int8),
        center=(9, 9),
        axes=(4, 6),
        angle=30,
        startAngle=0,
        endAngle=360,
        color=1,
        thickness=3,
    )

    actual = get_contours(labels)
    assert len(actual) == 2

    for c in actual:
        assert (labels[tuple(c)] == 1).all()


def test_interpolate_contour():
    import numpy as np

    from strainmap.models.ai_segmenter import interpolate_contour

    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    contour = np.array([np.cos(theta), np.sin(theta)])
    theta = np.linspace(0, 2 * np.pi, 21)
    expected = np.array([np.cos(theta), np.sin(theta)])
    actual = interpolate_contour(contour, len(theta))
    assert actual == pytest.approx(expected, rel=1e-1)


def test_labels_to_contours():
    import cv2
    import numpy as np

    from strainmap.models.ai_segmenter import labels_to_contours

    points = 361
    frames = 5
    labels = np.zeros((frames, 20, 20), dtype=np.int8)
    expected_shape = (2, frames, 2, points)
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels, points=points)
    assert actual.shape == expected_shape
    assert np.isnan(actual).all()
    assert record[-1].category == RuntimeWarning
    assert record[-1].message.args[0] == "Contours not found for 5 images."

    labels = np.array(
        [
            cv2.ellipse(
                img=np.zeros((20, 20), dtype=np.int8),
                center=(9, 9),
                axes=(4, 6),
                angle=30,
                startAngle=0,
                endAngle=360,
                color=1,
                thickness=2,
            )
            for _ in range(frames)
        ]
    )
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels, points=points)
    assert actual.shape == expected_shape
    assert not np.isnan(actual).any()
    assert len(record) == 0

    labels[0] = 0
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels, points=points)
    assert actual.shape == expected_shape
    assert np.isnan(actual[:, 0]).all()
    assert record[-1].category == RuntimeWarning
    assert record[-1].message.args[0] == "Contours not found for 1 images."


class TestNormal:
    def test_avail(self):
        from strainmap.models.ai_segmenter import Normal

        assert len(Normal.avail()) >= 1

    def test_register(self):
        from strainmap.models.ai_segmenter import Normal

        @Normal.register
        def my_norm():
            pass

        assert "my_norm" in Normal.avail()
        del Normal._normalizers["my_norm"]

    def test_run(self):
        import numpy as np

        from strainmap.models.ai_segmenter import Normal

        fun = MagicMock(__name__="my_norm")
        Normal.register(fun)
        data = np.array([1, 2, 3])
        Normal.run(data, "my_norm")
        fun.assert_called_with(data)
        del Normal._normalizers["my_norm"]


class TestUNet:
    def test_factory(self, data_shape, keras_model, tmp_path):
        import os

        import numpy as np

        from strainmap.models.ai_segmenter import UNet

        os.environ["STRAINMAP_AI_MODEL"] = ""
        with pytest.raises(RuntimeError):
            UNet.factory()

        keras_model.save(tmp_path)
        os.environ["STRAINMAP_AI_MODEL"] = str(tmp_path)
        loaded = UNet.factory()
        assert id(loaded) == id(UNet.factory())

        UNet._unet = None
        loaded = UNet.factory(tmp_path)
        assert id(loaded) == id(UNet.factory())

        data = np.random.random((10, *data_shape))
        np.testing.assert_allclose(
            keras_model.predict(data), loaded.model.predict(data)
        )

    def test_predict(self, data_shape):
        import numpy as np

        from strainmap.models.ai_segmenter import UNet

        net = UNet.factory()
        data = np.random.random((5, *data_shape))
        predicted = net.predict(data)
        assert predicted.shape == (5, data_shape[0], data_shape[1])
        assert predicted.dtype == np.int8
        assert all([v in (0, 1) for v in np.unique(predicted)])


def test_ai_segmentation(data_shape):
    import numpy as np
    import xarray as xr

    from strainmap.models.ai_segmenter import ai_segmentation

    points = 360
    n = 5
    data = xr.DataArray(
        np.random.random((n, *data_shape)), dims=["frame", "row", "col", "comp"]
    )
    contours = ai_segmentation(data, points=points)
    expected_shape = (2, n, 2, points)
    assert contours.transpose("side", "frame", "coord", "point").shape == expected_shape

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def data_shape():
    return 32


@pytest.fixture
def keras_model(data_shape):
    from tensorflow import keras
    import numpy as np

    train_input = np.random.random((128, data_shape))
    train_target = np.random.random((128, 1))

    inputs = keras.Input(shape=(data_shape,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(train_input, train_target)
    return model


def test_add_ellipse():
    from strainmap.models.ai_segmenter import add_ellipse
    import cv2
    import numpy as np

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
    from strainmap.models.ai_segmenter import get_contours
    import cv2
    import numpy as np

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
    from strainmap.models.ai_segmenter import interpolate_contour
    import numpy as np

    theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    contour = np.array([np.cos(theta), np.sin(theta)])
    theta = np.linspace(0, 2 * np.pi, 21)
    expected = np.array([np.cos(theta), np.sin(theta)])
    actual = interpolate_contour(contour, len(theta))
    assert actual == pytest.approx(expected, rel=1e-1)


def test_labels_to_contours():
    from strainmap.models.ai_segmenter import labels_to_contours
    import cv2
    import numpy as np

    points = 361
    labels = np.zeros((3, 20, 20), dtype=np.int8)
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels, points=points)
    assert actual.shape == (3, 2, 2, points)
    assert np.isnan(actual).all()
    assert record[-1].category == RuntimeWarning
    assert record[-1].message.args[0] == "Contours not found for 3 images."

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
            for i in range(3)
        ]
    )
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels)
    assert actual.shape == (3, 2, 2, points)
    assert not np.isnan(actual).any()
    assert len(record) == 0

    labels[0] = 0
    with pytest.warns(None) as record:
        actual = labels_to_contours(labels)
    assert actual.shape == (3, 2, 2, points)
    assert np.isnan(actual[0]).all()
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
        from strainmap.models.ai_segmenter import Normal

        fun = MagicMock(__name__="my_norm")
        Normal.register(fun)
        data = [1, 2, 3]
        Normal.run(data, "my_norm")
        fun.assert_called_with(data)
        del Normal._normalizers["my_norm"]


class TestUNet:
    def test_factory(self, data_shape, keras_model, tmp_path):
        from strainmap.models.ai_segmenter import UNet
        import numpy as np

        keras_model.save(tmp_path)
        loaded = UNet.factory(tmp_path)
        assert id(loaded) == id(UNet.factory())

        data = np.random.random((100, data_shape))
        np.testing.assert_allclose(
            keras_model.predict(data), loaded.model.predict(data)
        )

    def test_predict(self, data_shape):
        from strainmap.models.ai_segmenter import UNet
        import numpy as np

        net = UNet.factory()
        data = np.random.random((5, data_shape))
        predicted = net.predict(data)
        assert predicted.dtype == np.int8
        assert set(predicted) == {0, 1}

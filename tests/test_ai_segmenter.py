import pytest
from unittest.mock import MagicMock


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
    @pytest.mark.xfail
    def test_factory(self):
        assert False

    @pytest.mark.xfail
    def test_model(self):
        assert False

    @pytest.mark.xfail
    def test__conv_block(self):
        assert False

    @pytest.mark.xfail
    def test__deconv_block(self):
        assert False

    @pytest.mark.xfail
    def test__modelstruct(self):
        assert False

    @pytest.mark.xfail
    def test_compile_model(self):
        assert False

    @pytest.mark.xfail
    def test_train(self):
        assert False

    @pytest.mark.xfail
    def test_infer(self):
        assert False

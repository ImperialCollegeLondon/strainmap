import pytest
from unittest.mock import MagicMock


def test_zero2one():
    from strainmap.models.ai_segmenter import zero2one
    import numpy as np

    data = np.random.random((5, 10, 3))
    norm = zero2one(data)
    assert norm.min() == pytest.approx(0)
    assert norm.max() == pytest.approx(1)


def test_zeromean_unitvar():
    from strainmap.models.ai_segmenter import zeromean_unitvar
    import numpy as np

    data = np.random.random((5, 10, 3))
    norm = zeromean_unitvar(data)
    assert np.mean(norm) == pytest.approx(0)
    assert np.std(norm) == pytest.approx(1)


def test_crop_roi():
    from strainmap.models.ai_segmenter import crop_roi
    import numpy as np

    labels = np.ones((20, 20))
    cropped = crop_roi(labels, margin=5)
    expected = np.zeros((20, 20), dtype=np.int8)
    expected[4:15, 4:15] = 1
    assert (cropped == expected).all()


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
        assert (labels[tuple(c[::-1])] == 1).all()


@pytest.mark.xfail
def test_interpolate_contour():
    assert False


@pytest.mark.xfail
def test_labels_to_contours():
    assert False


class TestDataAugmentation:
    @pytest.mark.xfail
    def test_factory(self):
        assert False

    @pytest.mark.xfail
    def test_transform(self):
        assert False

    @pytest.mark.xfail
    def test_augment(self):
        assert False

    @pytest.mark.xfail
    def test__group(self):
        assert False

    @pytest.mark.xfail
    def test__ungroup(self):
        assert False


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

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


class TestDataAugmentation:
    def test_factory(self):
        from strainmap.models.ai_segmenter import DataAugmentation
        import toml
        from pathlib import Path

        filename = (
            Path(__file__).parent.parent / "strainmap" / "models" / "ai_config.toml"
        )
        config = toml.load(filename)["augmentation"]

        da = DataAugmentation.factory()
        assert len(da.steps) == len([c for c in config["active"]])
        assert da.times == config["times"]
        assert da.axis == config["axis"]
        assert da.include_original == config["include_original"]

    def test_transform(self):
        from strainmap.models.ai_segmenter import DataAugmentation
        import numpy as np

        def double(d):
            return 2 * d

        n = 3
        da = DataAugmentation.factory()
        da.steps = [
            double,
        ] * n
        data = np.random.random((5, 5))
        actual = da.transform(data)
        assert actual == pytest.approx(data * 2 ** n)

    def test_augment(self):
        from strainmap.models.ai_segmenter import DataAugmentation
        import numpy as np

        def double(d):
            return 2 * d

        n = 10
        c = 4
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        da = DataAugmentation.factory()
        da.steps = [
            double,
        ]

        da.include_original = False
        aug_img, aug_lbl = da.augment(images, labels)
        assert aug_img.shape == (n * da.times, h, w, c)
        assert aug_lbl.shape == (n * da.times, h, w)

        da.include_original = True
        aug_img, aug_lbl = da.augment(images, labels)
        assert aug_img.shape == (n * (da.times + 1), h, w, c)
        assert aug_lbl.shape == (n * (da.times + 1), h, w)

    def test__group(self):
        from strainmap.models.ai_segmenter import DataAugmentation
        import numpy as np

        n = 3
        c = 2
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        grouped = DataAugmentation._group(images, labels)
        assert grouped.shape == (n * (c + 1), h, w)
        for i in range(n):
            assert (grouped[i : c * n : n].transpose((1, 2, 0)) == images[i]).all()
        assert (grouped[-n:] == labels).all()

    def test__ungroup(self):
        from strainmap.models.ai_segmenter import DataAugmentation
        import numpy as np

        n = 3
        c = 2
        h = w = 5
        images = np.random.random((n, h, w, c))
        labels = np.random.random((n, h, w))
        grouped = DataAugmentation._group(images, labels)
        eimages, elabels = DataAugmentation._ungroup(grouped, c)
        assert (eimages == images).all()
        assert (elabels == labels).all()


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

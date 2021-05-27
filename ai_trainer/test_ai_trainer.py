import pytest
from unittest.mock import MagicMock


class TestDataAugmentation:
    def test_factory(self):
        from .unet import DataAugmentation
        import toml
        from pathlib import Path

        filename = Path(__file__).parent / "ai_config.toml"
        config = toml.load(filename)["augmentation"]

        da = DataAugmentation.factory()
        assert len(da.steps) == len([c for c in config["active"]])
        assert da.times == config["times"]
        assert da.axis == config["axis"]
        assert da.include_original == config["include_original"]

    def test_transform(self):
        from .unet import DataAugmentation
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
        from .unet import DataAugmentation
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
        from .unet import DataAugmentation
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
        from .unet import DataAugmentation
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
        from .unet import Normal

        assert len(Normal.avail()) >= 1

    def test_register(self):
        from .unet import Normal

        @Normal.register
        def my_norm():
            pass

        assert "my_norm" in Normal.avail()
        del Normal._normalizers["my_norm"]

    def test_run(self):
        from .unet import Normal

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

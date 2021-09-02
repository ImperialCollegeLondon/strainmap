from pytest import approx


def test_read_images(data_tree):
    from random import choice
    import pydicom
    from strainmap.models.readers import read_images

    cine = choice(data_tree.cine.values)
    comp = choice(data_tree.raw_comp.values)
    images = read_images(data_tree.sel(cine=cine, raw_comp=comp).values)
    assert len(images) == data_tree.sizes["frame"]

    for i, im in enumerate(images):
        data = pydicom.dcmread(data_tree.sel(cine=cine, raw_comp=comp)[i].item())
        columns = data.Columns
        rows = data.Rows
        assert im.shape == (rows, columns)


def test_velocity_sensitivity(dicom_data_path):
    from strainmap.models.readers import velocity_sensitivity
    from strainmap.models.strainmap_data_model import StrainMapData
    import numpy as np
    import pydicom

    data_tree = StrainMapData.from_folder(data_files=dicom_data_path).data_files.files
    filename = data_tree[0, 0, 0].item()

    expected = np.array((40, 40, 60))
    ds = pydicom.dcmread(filename)
    header = ds[("0021", "1019")].value.decode()

    actual = velocity_sensitivity(header)
    assert expected == approx(actual.data)


def test_dicom_reader():
    from strainmap.models.readers import DICOM
    from pathlib import Path

    path = Path(__file__).parent / "data" / "CM1"
    assert DICOM.belongs(path)

    files = DICOM.factory(path)
    assert files.is_avail
    assert len(files.datasets) == 5
    assert list(files.sensitivity) == [40.0, 40.0, 60]
    swap, signs = files.phase_encoding(files.datasets[0])
    assert not swap
    assert list(signs) == [1, 1, 1]
    assert files.tags(files.datasets[0])["PatientName"] == "CM"
    assert files.mag(files.datasets[0]).shape == (3, 512, 512)
    assert files.phase(files.datasets[0]).shape == (3, 3, 512, 512)
    assert files.cine_loc(files.datasets[0])
    assert files.pixel_size(files.datasets[0])
    assert files.time_interval(files.datasets[0])


def test_readers_registry():
    from strainmap.models.readers import (
        DICOM_READERS,
        register_dicom_reader,
        DICOMReaderBase,
    )

    assert len(DICOM_READERS) > 0

    class Dummy:
        pass

    register_dicom_reader(Dummy)
    assert all([issubclass(c, DICOMReaderBase) for c in DICOM_READERS])


def test_read_netcdf_file(tmp_path):
    from strainmap.models.writers import write_netcdf_file
    from strainmap.models.readers import read_netcdf_file
    import xarray as xr
    import numpy as np

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 5)), "bar": (("x"), np.random.rand(4))},
        coords={
            "x": [10, 20, 30, 40],
            "y": np.linspace(0, 10, 5),
            "z": ("y", list("abcde")),
        },
    )

    filename = tmp_path / "my_data.nc"
    write_netcdf_file(filename, foo=ds.foo, bar=ds.bar, patient="Thor", age=38)

    data = read_netcdf_file(filename)
    assert data["filename"] == filename
    assert data["patient"] == "Thor"
    assert data["age"] == 38
    xr.testing.assert_allclose(data["foo"], ds.foo)
    xr.testing.assert_allclose(data["bar"], ds.bar)

from unittest.mock import MagicMock, patch

import pytest
from pytest import raises


def test_save_group(tmp_path):
    from strainmap.models.writers import save_group

    to_netcdf = MagicMock()

    with patch("xarray.DataArray.to_netcdf", to_netcdf):
        import xarray as xr

        da = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": ["a", "b", "c"]})

        filename = tmp_path / "my_data.nc"
        save_group(filename, da, "speed")
        to_netcdf.assert_called_with(filename, mode="w", group="speed", encoding={})

        to_netcdf.reset_mock()
        filename.open("w").close()
        save_group(filename, da, "speed")
        to_netcdf.assert_called_with(filename, mode="a", group="speed", encoding={})

        to_netcdf.reset_mock()
        save_group(filename, da, "speed", overwrite=True)
        to_netcdf.assert_called_with(filename, mode="w", group="speed", encoding={})

        to_netcdf.reset_mock()
        save_group(filename, da, "speed", to_int=True)
        assert to_netcdf.call_args[-1]["encoding"] != {}


def test_save_attribute(tmp_path):
    import h5py

    from strainmap.models.writers import save_attribute

    filename = tmp_path / "my_data.nc"
    save_attribute(filename, name="Thor")

    f = h5py.File(filename, mode="r")
    assert f.attrs["name"] == "Thor"


def test_write_netcdf_file(tmp_path):
    import h5py
    import numpy as np
    import xarray as xr

    from strainmap.models.writers import write_netcdf_file

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

    data = h5py.File(filename, mode="r")
    assert data.attrs["patient"] == "Thor"
    assert data.attrs["age"] == 38

    for var in (ds.foo, ds.bar):
        assert var.name in data
        assert var.name in data[var.name]
        for coord in var.coords:
            assert coord in data[var.name]


@pytest.mark.xfail(reason="h5repack seems to be missing in most installations")
def test_repack_file(tmp_path):
    import xarray as xr

    from strainmap.models.writers import repack_file, write_netcdf_file

    da = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": ["a", "b", "c"]})

    filename = tmp_path / "my_data.nc"

    # Write the file twice to create some space issues
    write_netcdf_file(filename, foo=da)
    write_netcdf_file(filename, foo=da)

    repack_file(filename)
    target = filename.parent / f"~{filename.name}"
    assert target.is_file()
    assert target.stat().st_size <= filename.stat().st_size

    with raises(RuntimeError):
        repack_file(filename, target)

    target = filename.parent / "new_filename.nc"
    repack_file(filename, target)
    assert target.is_file()

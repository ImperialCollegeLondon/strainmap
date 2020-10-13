import sys

from pytest import approx, mark


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

    expected = np.array((60, 40, 40))
    ds = pydicom.dcmread(filename)
    header = ds[("0021", "1019")].value.decode()

    actual = velocity_sensitivity(header)
    assert expected == approx(actual.data)


def test_read_data_structure(tmpdir, segmented_data):
    from strainmap.models.readers import read_data_structure
    from strainmap.models.writers import write_data_structure
    from collections import defaultdict
    import h5py

    filename = tmpdir / "strain_map_file.h5"
    f = h5py.File(filename, "a")

    cine = segmented_data.segments.cine[0].item()
    write_data_structure(f, "segments", segmented_data.segments)

    d = defaultdict(dict)
    read_data_structure(d, f["segments"])

    assert all([key in d for key in f["segments"]])
    assert all([value.keys() == d[key].keys() for key, value in f["segments"].items()])
    assert d[cine]["endocardium"] == approx(f["segments"][cine]["endocardium"][...])


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_from_relative_paths(tmpdir):
    from strainmap.models.readers import from_relative_paths
    from pathlib import Path

    master = Path("home/data/my_file.h5").resolve()
    expected = [
        str(Path("home").resolve()),
        str(Path("home/data").resolve()),
        str(Path("home/data/cars").resolve()),
        str(Path("home/data/cars/Tesla").resolve()),
    ]
    paths = [b"..", b".", b"cars", b"cars/Tesla"]
    actual = from_relative_paths(master, paths)
    assert actual == expected


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_paths_from_hdf5(strainmap_data, tmpdir):
    from strainmap.models.writers import paths_to_hdf5
    from strainmap.models.readers import paths_from_hdf5
    from collections import defaultdict
    from strainmap.coordinates import Comp

    import h5py

    mag = strainmap_data.data_files.vars[Comp.MAG]
    cine = strainmap_data.data_files.datasets[0]
    filename = tmpdir / "strain_map_file.h5"

    abs_paths = strainmap_data.data_files.files.isel(cine=0, raw_comp=0).values

    f = h5py.File(filename, "a")
    d = defaultdict(dict)
    paths_to_hdf5(f, filename, strainmap_data.data_files.files)
    paths_from_hdf5(d, filename, f["data_files"])

    assert cine in d
    assert all(
        [key in d[cine] for key in strainmap_data.data_files.files.raw_comp.values]
    )
    assert (d[cine][mag] == abs_paths).all()


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_read_h5_file(tmpdir, segmented_data):
    from strainmap.models.readers import read_h5_file
    from strainmap.models.writers import write_hdf5_file
    from strainmap.models.strainmap_data_model import StrainMapData

    filename = tmpdir / "strain_map_file.h5"

    write_hdf5_file(segmented_data, filename)
    attributes = read_h5_file(StrainMapData.stored, filename)
    new_data = StrainMapData.from_folder()
    new_data.__dict__.update(attributes)

    assert all(segmented_data == new_data)


def test_dicom_reader():
    from strainmap.models.readers import DICOM
    from pathlib import Path

    path = Path(__file__).parent / "data" / "CM1"
    assert DICOM.belongs(path)

    files = DICOM.factory(path)
    assert files.is_avail
    assert len(files.datasets) == 5
    assert list(files.sensitivity) == [60.0, 40.0, 40.0]
    swap, signs = files.phase_encoding(files.datasets[0])
    assert not swap
    assert list(signs) == [1, -1, 1]
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


def test_labels_to_group(larray, tmpdir):
    from strainmap.models.writers import labels_to_group
    from strainmap.models.readers import labels_from_group
    import h5py

    filename = tmpdir / "strainmap_file.h5"
    f = h5py.File(filename, "a")
    labels_to_group(
        f.create_group("labels", track_order=True), larray.dims, larray.coords
    )
    dims, coords = labels_from_group(f["labels"])

    assert dims == larray.dims
    assert coords == larray.coords


def test_group_to_labelled_array(larray, tmpdir):
    from strainmap.models.writers import labelled_array_to_group
    from strainmap.models.readers import group_to_labelled_array
    import h5py

    filename = tmpdir / "strainmap_file.h5"
    f = h5py.File(filename, "a")
    labelled_array_to_group(f.create_group("array"), larray)
    new_larray = group_to_labelled_array(f["array"])

    assert new_larray == larray

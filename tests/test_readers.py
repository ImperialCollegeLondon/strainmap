from pathlib import Path
from typing import Mapping, Text, Tuple, Union
from pytest import approx, raises


def search_in_tree(
    tree: Mapping, filename: Union[Path, Text]
) -> Tuple[Text, Text, int]:

    for s in tree:
        for v in tree[s]:
            if filename in tree[s][v]:
                return s, v, tree[s][v].index(filename)

    raise ValueError(f"Could not find file in tree {filename}")


def test_read_dicom_directory_tree(dicom_data_path):
    from strainmap.models.readers import read_dicom_directory_tree, VAR_OFFSET

    data = read_dicom_directory_tree(dicom_data_path)

    assert isinstance(data, Mapping)
    assert len(data.keys()) == 3

    for k in data.keys():
        assert set(data[k].keys()) == set(VAR_OFFSET.keys())

        for v in data[k].keys():
            assert len(data[k][v]) == 3


def test_read_dicom_file_tags_from_file(dicom_data_path):
    from glob import glob
    from random import choice
    import pydicom
    from strainmap.models.readers import read_dicom_file_tags

    filename = choice(glob(str(dicom_data_path / "*.dcm")))

    actual = read_dicom_file_tags(filename)
    expected = pydicom.dcmread(filename)

    for d in expected.dir():
        assert actual[d] == getattr(expected, d)


def test_read_dicom_file_tags_from_dict(dicom_data_path):
    from glob import glob
    from random import choice
    import pydicom
    from strainmap.models.readers import read_dicom_file_tags, read_dicom_directory_tree

    filename = choice(glob(str(dicom_data_path / "*.dcm")))
    data = read_dicom_directory_tree(dicom_data_path)

    series, variable, idx = search_in_tree(data, filename)

    actual = read_dicom_file_tags(data, series, variable, idx)
    expected = pydicom.dcmread(filename)

    for d in expected.dir():
        assert actual[d] == getattr(expected, d)


def test_read_images(data_tree):
    from random import choice
    from strainmap.models.readers import read_dicom_file_tags, read_images

    series = choice(list(data_tree.keys()))
    variable = choice(list(data_tree[series].keys()))
    images = read_images(data_tree, series, variable)
    assert len(images) == len(data_tree[series][variable])

    for i, im in enumerate(images):
        dicom_info = read_dicom_file_tags(data_tree[series][variable][i])
        columns = dicom_info["Columns"]
        rows = dicom_info["Rows"]
        assert im.shape == (rows, columns)


def test_read_all_images(data_tree):
    from strainmap.models.readers import read_all_images
    import numpy as np

    images = read_all_images(data_tree)

    for s in data_tree.keys():
        assert s in images.keys()
        for v in data_tree[s].keys():
            assert v in images[s].keys()
            for i in images[s][v]:
                assert isinstance(i, np.ndarray)


def test_to_numpy(data_tree):
    from strainmap.models.readers import images_to_numpy, read_all_images

    data = images_to_numpy(read_all_images(data_tree))
    assert set(data.keys()) == set(data_tree.keys())

    magnitude, phase = data[list(data.keys())[0]]
    assert magnitude.shape == (3, 3, 512, 512)
    assert phase.shape == (3, 3, 512, 512)


def test_velocity_sensitivity(data_tree):
    from strainmap.models.readers import velocity_sensitivity
    import numpy as np

    filename = list(data_tree.values())[0]["PhaseZ"][0]
    expected = np.array((60, 40, 40))
    actual = velocity_sensitivity(filename)

    assert expected == approx(actual)


def test_read_data_structure(tmpdir, segmented_data):
    from strainmap.models.readers import read_data_structure
    from strainmap.models.writers import write_data_structure
    from collections import defaultdict
    import h5py

    filename = tmpdir / "strain_map_file.h5"
    f = h5py.File(filename, "a")

    dataset_name = list(segmented_data.segments.keys())[0]
    write_data_structure(f, "segments", segmented_data.segments)

    d = defaultdict(dict)
    read_data_structure(d, f["segments"])

    assert all([key in d for key in f["segments"]])
    assert all([value.keys() == d[key].keys() for key, value in f["segments"].items()])
    assert d[dataset_name]["endocardium"] == approx(
        f["segments"][dataset_name]["endocardium"][...]
    )


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


def test_paths_from_hdf5(strainmap_data, tmpdir):
    from strainmap.models.writers import paths_to_hdf5
    from strainmap.models.readers import paths_from_hdf5
    from collections import defaultdict

    import h5py

    dataset_name = list(strainmap_data.data_files.keys())[0]
    filename = tmpdir / "strain_map_file.h5"

    abs_paths = strainmap_data.data_files[dataset_name]["MagX"]

    f = h5py.File(filename, "a")
    d = defaultdict(dict)
    paths_to_hdf5(f, filename, "data_files", strainmap_data.data_files)
    paths_from_hdf5(d, filename, f["data_files"])

    assert dataset_name in d
    assert all(
        [key in d[dataset_name] for key in strainmap_data.data_files[dataset_name]]
    )
    if str(filename)[0] != abs_paths[0][0]:
        assert d[dataset_name]["MagX"] == []
    else:
        assert d[dataset_name]["MagX"] == abs_paths


def test_read_h5_file(tmpdir, segmented_data):
    from strainmap.models.readers import read_h5_file
    from strainmap.models.writers import write_hdf5_file
    from strainmap.models.strainmap_data_model import StrainMapData

    filename = tmpdir / "strain_map_file.h5"
    dataset_name = list(segmented_data.data_files.keys())[0]
    abs_paths = segmented_data.data_files[dataset_name]["MagX"]

    write_hdf5_file(segmented_data, filename)
    new_data = read_h5_file(StrainMapData.from_folder(), filename)

    if str(filename)[0] != abs_paths[0][0]:
        assert segmented_data != new_data
    else:
        assert segmented_data == new_data


def test_legacy_dicom():
    from strainmap.models.readers import LegacyDICOM
    from pathlib import Path

    path = Path(__file__).parent / "data" / "SUB1"
    assert LegacyDICOM.belongs(path)

    files = LegacyDICOM.factory(path)
    assert files.is_avail
    assert len(files.datasets) == 3
    assert list(files.sensitivity) == [60.0, 40.0, 40.0]
    swap, signs = files.orientation
    assert not swap
    assert list(signs) == [1, -1, 1]
    assert files.tags(files.datasets[0])["PatientName"] == "SUBJECT1"
    assert files.mag(files.datasets[0]).shape == (3, 512, 512)
    assert files.phase(files.datasets[0]).shape == (3, 3, 512, 512)
    with raises(AttributeError):
        assert files.slice_loc(files.datasets[0])
    with raises(AttributeError):
        assert files.pixel_size(files.datasets[0])
    with raises(AttributeError):
        files.time_interval(files.datasets[0])


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

from pathlib import Path
from typing import Mapping, Text, Tuple, Union
from pytest import approx
import numpy as np


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
    expected = np.array((20, 20, 30))
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
    assert d[dataset_name]["MagX"] == abs_paths


def compare_dicts(one, two):
    """Recursive comparison of two (nested) dictionaries with lists and numpy arrays."""
    if one.keys() != two.keys():
        return False

    equal = True
    for key, value in one.items():
        if isinstance(value, dict) and isinstance(two[key], dict):
            equal = compare_dicts(value, two[key]) and equal
        elif not isinstance(value, dict) and not isinstance(two[key], dict):
            equal = (np.array(value) == np.array(two[key])).all() and equal
        else:
            return False

        if not equal:
            return False

    return True


def test_read_h5_file(tmpdir, segmented_data):
    from strainmap.models.readers import read_h5_file
    from strainmap.models.writers import write_hdf5_file

    filename = tmpdir / "strain_map_file.h5"

    write_hdf5_file(segmented_data, filename)
    new_data = read_h5_file(filename)

    for s in new_data.__dict__.keys():
        if s == "strainmap_file":
            continue
        else:
            assert compare_dicts(getattr(new_data, s), getattr(segmented_data, s))

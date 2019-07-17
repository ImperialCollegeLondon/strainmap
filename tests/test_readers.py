from pathlib import Path
from typing import Mapping, Text, Tuple, Union


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

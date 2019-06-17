from typing import Mapping, Tuple, Text, Union, Optional
from pathlib import Path


def search_in_tree(
    tree: Mapping, filename: Union[Path, Text]
) -> Optional[Tuple[Text, Text, int]]:

    for s in tree:
        for v in tree[s]:
            if filename in tree[s][v]:
                return s, v, tree[s][v].index(filename)

    return None


def test_read_dicom_directory_tree(dicom_data_path):
    from strainmap.model.readers import read_dicom_directory_tree, VAR_OFFSET

    data = read_dicom_directory_tree(dicom_data_path)

    assert isinstance(data, Mapping)
    assert len(data.keys()) == 3

    for k in data.keys():
        assert set(data[k].keys()) == set(VAR_OFFSET.keys())

        for v in data[k].keys():
            assert len(data[k][v]) == 50


def test_read_dicom_file_tags_from_file(dicom_data_path):
    from glob import glob
    from random import choice
    import pydicom
    from strainmap.model.readers import read_dicom_file_tags

    filename = choice(glob(str(dicom_data_path / "*.dcm")))

    actual = read_dicom_file_tags(filename)
    expected = pydicom.dcmread(filename)

    for d in expected.dir():
        assert actual[d] == getattr(expected, d)


def test_read_dicom_file_tags_from_dict(dicom_data_path):
    from glob import glob
    from random import choice
    import pydicom
    from strainmap.model.readers import read_dicom_file_tags, read_dicom_directory_tree

    filename = choice(glob(str(dicom_data_path / "*.dcm")))
    data = read_dicom_directory_tree(dicom_data_path)

    series, variable, idx = search_in_tree(data, filename)

    actual = read_dicom_file_tags(data, series, variable, idx)
    expected = pydicom.dcmread(filename)

    for d in expected.dir():
        assert actual[d] == getattr(expected, d)


def test_read_images(dicom_data_path):
    from random import choice
    from strainmap.model.readers import (
        read_dicom_directory_tree,
        read_dicom_file_tags,
        read_images,
    )

    data = read_dicom_directory_tree(dicom_data_path)
    series = choice(list(data.keys()))
    variable = choice(list(data[series].keys()))
    images = read_images(data, series, variable)
    assert len(images) == len(data[series][variable])

    for i, im in enumerate(images):
        dicom_info = read_dicom_file_tags(data[series][variable][i])
        columns = dicom_info["Columns"]
        rows = dicom_info["Rows"]
        assert im.shape == (rows, columns)

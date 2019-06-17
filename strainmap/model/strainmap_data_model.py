from pathlib import Path
from typing import Mapping, Optional, Text, Union

from .readers import read_dicom_directory_tree, read_dicom_file_tags, read_images


class StrainMapData(object):
    def __init__(self, data_files: Mapping, bg_files: Optional[Mapping] = None):

        self.data_files = data_files
        self.bg_files = bg_files if bg_files else {}

    def read_dicom_file_tags(self, series, variable, idx):
        return read_dicom_file_tags(self.data_files, series, variable, idx)

    def get_images(self, series, variable):
        return read_images(self.data_files, series, variable)


def factory(
    data: Optional[StrainMapData] = None,
    data_files: Union[Path, Text, None] = None,
    bg_files: Union[Path, Text, None] = None,
    strainmap_file: Union[Path, Text, None] = None,
):
    """ Creates a new StrainMapData object or updates the data of an existing one.

    The exiting object might be passed as an argument or be loaded from an HDF5 or
    Matlab files. In either case, its data_files, bg_files or both will be updated
    accordingly.

    If there is no existing object, a new one will be created from the data_files and
    the bg_files.
    """
    if data_files:
        data_files = read_dicom_directory_tree(data_files)

    if bg_files:
        bg_files = read_dicom_directory_tree(bg_files)

    if strainmap_file is not None:
        # TODO Placeholder for loading objects from HDF5 and Matlab files
        pass

    if isinstance(data, StrainMapData):
        data.data_files = data_files if data_files else data.data_files
        data.bg_files = bg_files if bg_files else data.bg_files
    elif data_files:
        data = StrainMapData(data_files=data_files, bg_files=bg_files)
    else:
        raise RuntimeError(
            "Insufficient information to create or update a StrainMapData object."
        )

    return data

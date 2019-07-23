from pathlib import Path
from typing import Mapping, Optional, Text, Union

from .readers import read_dicom_directory_tree, read_dicom_file_tags, read_images


class StrainMapLoadError(Exception):
    pass


class StrainMapData(object):
    def __init__(self, data_files: Mapping, bg_files: Optional[Mapping] = None):

        self.data_files = data_files
        self.bg_files = bg_files if bg_files else {}
        self.segments: dict = {}
        self.velocities: dict = {}

    def read_dicom_file_tags(self, series, variable, idx):
        return read_dicom_file_tags(self.data_files, series, variable, idx)

    def get_images(self, series, variable):
        return read_images(self.data_files, series, variable)

    def get_bg_images(self, series, variable):
        return read_images(self.bg_files, series, variable)


def factory(
    data: Optional[StrainMapData] = None,
    data_files: Union[Path, Text, None] = None,
    bg_files: Union[Path, Text, None] = None,
    strainmap_file: Union[Path, Text, None] = None,
) -> StrainMapData:
    """ Creates a new StrainMapData object or updates the data of an existing one.

    The exiting object might be passed as an argument or be loaded from an HDF5 or
    Matlab files. In either case, its data_files, bg_files or both will be updated
    accordingly.

    If there is no existing object, a new one will be created from the data_files and
    the bg_files.
    """
    df: Optional[Mapping] = None
    bg: Optional[Mapping] = None
    if data_files is not None:
        df = read_dicom_directory_tree(data_files)

    if bg_files is not None:
        bg = read_dicom_directory_tree(bg_files)

    if strainmap_file is not None:
        # TODO Placeholder for loading objects from HDF5 and Matlab files
        pass

    if isinstance(data, StrainMapData):
        data.data_files = df if df is not None else data.data_files
        data.bg_files = bg if bg is not None else data.bg_files
    elif df:
        data = StrainMapData(data_files=df, bg_files=bg)
    else:
        data = StrainMapData(data_files={}, bg_files={})

    return data

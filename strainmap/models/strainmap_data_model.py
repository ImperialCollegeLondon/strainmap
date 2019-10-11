from pathlib import Path
from typing import Mapping, Optional, Text, Union
from collections import defaultdict
import numpy as np
import h5py

from .readers import (
    read_dicom_directory_tree,
    read_dicom_file_tags,
    read_images,
    read_strainmap_file,
)
from .writers import write_data_structure, write_hdf5_file


class StrainMapLoadError(Exception):
    pass


class StrainMapData(object):
    def __init__(
        self,
        data_files: Mapping,
        bg_files: Optional[Mapping] = None,
        strainmap_file: Optional[h5py.File] = None,
    ):

        self.data_files = data_files
        self.bg_files = bg_files if bg_files else defaultdict(dict)
        self.strainmap_file = strainmap_file
        self.segments: dict = defaultdict(dict)
        self.zero_angle: dict = {}
        self.velocities: dict = defaultdict(dict)
        self.masks: dict = defaultdict(dict)
        self.markers: dict = defaultdict(dict)

    def metadata(self, dataset=None):
        """Retrieve the metadata from the DICOM files"""
        if dataset is None:
            output = dict()
            dataset = list(self.data_files.keys())[0]
        else:
            output = {"Dataset": dataset}

        patient_data = self.read_dicom_file_tags(dataset, "MagZ", 0)
        output.update(
            {
                "Patient Name": str(patient_data.get("PatientName", "")),
                "Patient DOB": str(patient_data.get("PatientBirthDate", "")),
                "Date of Scan": str(patient_data.get("StudyDate", "")),
            }
        )
        return output

    def read_dicom_file_tags(self, series, variable, idx):
        return read_dicom_file_tags(self.data_files, series, variable, idx)

    def get_images(self, series, variable):
        return np.array(read_images(self.data_files, series, variable))

    def get_bg_images(self, series, variable):
        return np.array(read_images(self.bg_files, series, variable))

    def save(self, structure: Union[list, str, None] = None):
        """Saves the data to the hdf5 file, if present."""
        if self.strainmap_file is None:
            return
        elif structure is None:
            write_hdf5_file(self, self.strainmap_file)
        else:
            structure = [structure] if isinstance(structure, str) else structure
            for s in structure:
                write_data_structure(self.strainmap_file, s, getattr(self, s))


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
    if data_files is not None:
        df = read_dicom_directory_tree(data_files)

    bg: Optional[Mapping] = None
    if bg_files is not None:
        bg = read_dicom_directory_tree(bg_files)

    sm_file: Optional[h5py.File] = None
    if strainmap_file is not None:
        if Path(strainmap_file).is_file():
            data = read_strainmap_file(strainmap_file)
        elif str(strainmap_file).endswith(".h5"):
            sm_file = h5py.File(strainmap_file, "a")
        else:
            raise RuntimeError("File type cannot be opened by StrainMap.")

    if isinstance(data, StrainMapData):
        data.data_files = df if df is not None else data.data_files
        data.bg_files = bg if bg is not None else data.bg_files
        data.save()
    elif df:
        data = StrainMapData(data_files=df, bg_files=bg, strainmap_file=sm_file)
        data.save()
    else:
        data = StrainMapData(
            data_files=defaultdict(dict),
            bg_files=defaultdict(dict),
            strainmap_file=sm_file,
        )

    return data

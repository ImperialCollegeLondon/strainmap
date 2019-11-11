from pathlib import Path
from typing import Mapping, Optional, Text, Union, Tuple
from collections import defaultdict
import numpy as np
import h5py
from functools import reduce

from .readers import (
    read_dicom_directory_tree,
    read_dicom_file_tags,
    read_images,
    read_strainmap_file,
)
from .writers import write_hdf5_file


class StrainMapLoadError(Exception):
    pass


def compare_dicts(one, two):
    """Recursive comparison of two (nested) dictionaries with lists and numpy arrays."""
    if one.keys() != two.keys():
        return False

    equal = True
    for key, value in one.items():
        if isinstance(value, dict) and isinstance(two[key], dict):
            equal = compare_dicts(value, two[key]) and equal
        elif not isinstance(value, dict) and not isinstance(two[key], dict):
            if (
                len(value) != len(two[key])
                or not (np.array(value) == np.array(two[key])).all()
            ):
                return False
        else:
            return False

        if not equal:
            return False

    return True


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
        self.sign_reversal: Tuple[bool, ...] = (False, False, False)
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

    def save_all(self):
        """Saves the data to the hdf5 file, if present."""
        if self.strainmap_file is None:
            return

        write_hdf5_file(self, self.strainmap_file)

    def save(self, *args):
        """Saves specific datasets to the hdf5 file.

        Each dataset to be saved must be defined as a list of keys, where key[0] must be
        one of the StrainMapData attributes (segments, velocities, etc.)
        """
        if self.strainmap_file is None:
            return

        for keys in args:
            s = "/".join(keys)
            keys[0] = getattr(self, keys[0])
            if s in self.strainmap_file:
                self.strainmap_file[s][...] = reduce(lambda x, y: x[y], keys)
            else:
                self.strainmap_file.create_dataset(
                    s, data=reduce(lambda x, y: x[y], keys), track_order=True
                )

    def delete(self, *args):
        """ Deletes the chosen dataset or group from the hdf5 file.

        Each dataset to be saved must be defined as a list of keys, where key[0]
        must be one of the StrainMapData attributes (segments, velocities, etc.)
        """
        if self.strainmap_file is None:
            return

        for keys in args:
            s = "/".join(keys)
            if s in self.strainmap_file:
                del self.strainmap_file[s]

    def __eq__(self, other) -> bool:
        """Compares two StrainMapData objects.

        The "strainmap_file" attribute is ignored as it might have different values."""
        equal = self.sign_reversal == other.sign_reversal
        keys = set(self.__dict__.keys()) - {"sign_reversal", "strainmap_file"}
        for k in keys:
            equal = equal and compare_dicts(getattr(self, k), getattr(other, k))
        return equal


def factory(
    data: Optional[StrainMapData] = None,
    data_files: Union[Path, Text, None] = None,
    bg_files: Union[Path, Text, None] = None,
    strainmap_file: Union[Path, Text, None] = None,
) -> StrainMapData:
    """ Creates a new StrainMapData object or updates the data of an existing one.

    The exiting object might be passed as an argument or be loaded from an HDF5file.
    In either case, its data_files, bg_files or both will be updated accordingly.

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
        data.strainmap_file = sm_file if sm_file is not None else data.strainmap_file
        if data.strainmap_file is None:
            data.save_all()
    elif df:
        data = StrainMapData(data_files=df, bg_files=bg, strainmap_file=sm_file)
        data.save_all()
    else:
        data = StrainMapData(
            data_files=defaultdict(dict),
            bg_files=defaultdict(dict),
            strainmap_file=sm_file,
        )

    return data

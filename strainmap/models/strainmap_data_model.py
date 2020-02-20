from pathlib import Path
from typing import Optional, Text, Union, Tuple, Type
from collections import defaultdict
import numpy as np
import h5py
from functools import reduce

from .readers import read_strainmap_file, DICOMReaderBase, read_folder
from .writers import write_hdf5_file


class StrainMapLoadError(Exception):
    pass


def compare_dicts(one, two):
    """Recursive comparison of two (nested) dictionaries with lists and numpy arrays."""
    if not isinstance(one, dict):
        return True
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
    stored = (
        "data_files",
        "bg_files",
        "sign_reversal",
        "segments",
        "zero_angle",
        "markers",
    )

    @classmethod
    def from_folder(cls, data_files: Union[Path, Text, None] = None):
        """Creates a new StrainMap data object from a folder containing DICOMs"""
        data = cls()
        if data_files is not None:
            data = cls(data_files=read_folder(data_files))
            data.save_all()
        return data

    @classmethod
    def from_file(cls, strainmap_file: Union[Path, Text]):
        """Creates a new StrainMap data object from a h5 file.
        TODO Adapt to the new DICOM reader."""
        assert Path(strainmap_file).is_file()
        attributes = read_strainmap_file(cls.stored, strainmap_file)
        result = cls.from_folder()
        result.__dict__.update(attributes)
        result.regenerate()
        return result

    def __init__(
        self,
        data_files: Optional[Type[DICOMReaderBase]] = None,
        bg_files: Optional[Type[DICOMReaderBase]] = None,
        strainmap_file: Optional[h5py.File] = None,
    ):

        self.data_files = data_files
        self.bg_files = bg_files
        self.strainmap_file = strainmap_file
        self.sign_reversal: Tuple[bool, ...] = (False, False, False)
        self.segments: dict = defaultdict(dict)
        self.zero_angle: dict = {}
        self.velocities: dict = defaultdict(dict)
        self.masks: dict = defaultdict(dict)
        self.markers: dict = defaultdict(dict)

    @property
    def rebuilt(self):
        """Flag to indicate if data structure has been rebuilt after loading."""
        return all([k in self.velocities.keys() for k in self.markers.keys()])

    def add_paths(
        self,
        data_files: Union[Path, Text, None] = None,
        bg_files: Union[Path, Text, None] = None,
    ):
        """Adds data and/or phantom paths to the object."""
        if data_files is not None:
            self.data_files = read_folder(data_files)
        if bg_files is not None:
            self.bg_files = read_folder(bg_files)
        if data_files is not None or bg_files is not None:
            if not self.rebuilt:
                self.regenerate()
            self.save_all()
            return True
        return False

    def add_h5_file(self, strainmap_file: Union[Path, Text]):
        """Creates anew h5 file in the given path and add it to the structure."""
        if not str(strainmap_file).endswith(".h5"):
            return False
        self.strainmap_file = h5py.File(strainmap_file, "a")
        self.save_all()
        return True

    def regenerate(self):
        """Regenerate velocities and masks information after loading from h5 file."""
        from .velocities import calculate_velocities

        # If there is no data paths yet, we postpone the regeneration.
        if self.data_files == ():
            return

        for dataset, markers in self.markers.items():
            regions = dict()
            for k in markers.keys():
                info = k.split(" - ")
                if info[-1] not in regions.keys():
                    regions[info[-1]] = dict(angular_regions=[], radial_regions=[])
                if "global" in info[0]:
                    regions[info[-1]]["global_velocity"] = True
                else:
                    rtype, num = info[0].split(" x")
                    regions[info[-1]][f"{rtype}_regions"].append(int(num))

            for bg, region in regions.items():
                calculate_velocities(
                    data=self,
                    dataset_name=dataset,
                    bg=bg,
                    sign_reversal=self.sign_reversal,
                    init_markers=False,
                    **region,
                )

    def metadata(self, dataset=None):
        """Retrieve the metadata from the DICOM files"""
        if dataset is None:
            output = dict()
            dataset = self.data_files.datasets[0]
        else:
            output = {"Dataset": dataset}

        patient_data = self.data_files.tags(dataset)
        output.update(
            {
                "Patient Name": str(patient_data.get("PatientName", "")),
                "Patient DOB": str(patient_data.get("PatientBirthDate", "")),
                "Date of Scan": str(patient_data.get("StudyDate", "")),
            }
        )
        return output

    def save_all(self):
        """Saves the data to the hdf5 file, if present."""
        if self.strainmap_file is None or self.data_files is None:
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
            assert keys[0] in self.stored, f"{keys[0]} is not storable."
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
        keys = set(self.__dict__.keys())
        for k in keys:
            equal = equal and compare_dicts(getattr(self, k), getattr(other, k))
        return equal

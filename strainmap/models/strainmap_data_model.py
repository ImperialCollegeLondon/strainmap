from collections import defaultdict
from pathlib import Path
from typing import Optional, Text, Union

import h5py
import numpy as np
import xarray as xr

from ..coordinates import Comp
from .readers import DICOMReaderBase, read_folder, read_strainmap_file
from .sm_data import LabelledArray
from .writers import write_hdf5_file

TIMESHIFT = -0.045
"""Default timeshift"""


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
        "sign_reversal",
        "orientation",
        "segments",
        "septum",
        "markers",
        "strain_markers",
        "timeshift",
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
        """Creates a new StrainMap data object from a h5 file."""
        assert Path(strainmap_file).is_file()
        attributes = read_strainmap_file(cls.stored, strainmap_file)
        result = cls.from_folder()
        result.__dict__.update(attributes)
        result.regenerate()
        return result

    def __init__(
        self,
        data_files: Optional[DICOMReaderBase] = None,
        strainmap_file: Optional[h5py.File] = None,
    ):

        self.data_files = data_files
        self.strainmap_file = strainmap_file
        self.orientation: str = "CCW"
        self.timeshift: float = TIMESHIFT
        self.sign_reversal: xr.DataArray = xr.DataArray(
            [1, 1, 1], dims=["comp"], coords={"comp": [Comp.X, Comp.Y, Comp.Z]}
        )
        self.segments: xr.DataArray = xr.DataArray()
        self.centroid: xr.DataArray = xr.DataArray()
        self.septum: xr.DataArray = xr.DataArray()
        self.velocities: xr.DataArray = xr.DataArray()
        self.masks: xr.DataArray = xr.DataArray()
        self.markers: xr.DataArray = xr.DataArray()
        self.cylindrical: xr.DataArray = xr.DataArray()
        self.strain: dict = defaultdict(dict)
        self.strain_markers: dict = defaultdict(dict)
        self.gls: np.ndarray = np.array([])
        self.twist: Optional[LabelledArray] = None

    @property
    def rebuilt(self):
        """Flag to indicate if data structure has been rebuilt after loading."""
        return all([k in self.velocities.keys() for k in self.markers.keys()])

    def add_paths(self, data_files: Union[Path, Text, None] = None):
        """Adds data and/or phantom paths to the object."""
        if data_files is None:
            return False

        self.data_files = read_folder(data_files)
        if not self.rebuilt:
            self.regenerate()
        self.save_all()
        return True

    def add_h5_file(self, strainmap_file: Union[Path, Text]):
        """Creates anew h5 file in the given path and add it to the structure."""
        if not str(strainmap_file).endswith(".h5"):
            return False
        self.strainmap_file = h5py.File(strainmap_file, "a")
        self.save_all()
        return True

    def add_data(self, cine: str, **kwargs):
        """Adds new data to the attributes, expanding the 'cine' coordinate.

        After updating the data object, it is saved. if the attribute already has data
        for that cine, nothing is done. Update the attribute directly in that case, eg.:

            data.segmentation[{"cine": cine}][...] = new_data

        Args:
            cine (str): Name of the cine for which information is to be added.
            **kwargs: The attributes to update with the values they should take for this
                cine.

        Returns:
            None
        """
        for attr, value in kwargs.items():
            if getattr(self, attr).shape == ():
                setattr(self, attr, value.expand_dims(cine=[cine]).copy())
            elif cine not in getattr(self, attr).cine:
                setattr(
                    self,
                    attr,
                    xr.concat(
                        [getattr(self, attr), value.expand_dims(cine=[cine]).copy()],
                        dim="cine",
                    ),
                )
            else:
                getattr(self, attr).loc[{"cine": cine}] = value

        self.save(list(kwargs.keys()))

    def set_orientation(self, orientation):
        """Sets the angular regions orientation (CW or CCW) and saves the data"""
        self.orientation = orientation
        self.save(["orientation"])

    def regenerate(self):
        """We create placeholders for the velocities that were expected.

        The velocities and masks will be created at runtime, just when needed."""
        from .segmentation import _calc_effective_centroids, _calc_centroids

        # If there is no data paths yet, we postpone the regeneration.
        if self.data_files == ():
            return

        # TODO: Remove when consolidated. Regenerate the centroid
        for d in self.septum:
            try:
                raw_centroids = _calc_centroids(self.segments.sel(cine=d))
                self.septum[d][..., 1] = _calc_effective_centroids(
                    raw_centroids, window=3
                )

            except Exception as err:
                raise RuntimeError(
                    f"Error when regenerating COM for dataset '{d}'. "
                    f"Error message: {err}."
                )

        for dataset, markers in self.markers.items():
            for k in markers.keys():
                self.velocities[dataset][k] = None

            # TODO To remove! Hack to add the radial data on legacy h5 files.
            self.velocities[dataset]["radial x3"] = None

        # TODO: Remove when consolidated. Heal the septum mid-point
        default = None
        for dataset, za in self.septum.items():
            try:
                if np.isnan(za[..., 0]).any():
                    za[..., 0] = default
                    self.save(["zero_angle", dataset])
                else:
                    default = za[..., 0].copy()

            except Exception as err:
                raise RuntimeError(
                    f"Error when healing septum mid-point for "
                    f"dataset '{dataset}'. Error message: {err}."
                )

    def metadata(self, dataset=None):
        """Retrieve the metadata from the DICOM files"""
        if dataset is None:
            output = dict()
            dataset = self.data_files.datasets[0]
        else:
            output = {"Cine": dataset}

        patient_data = self.data_files.tags(dataset)
        output.update(
            {
                "Patient Name": str(patient_data.get("PatientName", "")),
                "Patient DOB": str(patient_data.get("PatientBirthDate", "")),
                "Date of Scan": str(patient_data.get("StudyDate", "")),
                "Cine offset (mm)": str(self.data_files.cine_loc(dataset) * 10),
                "Mean RR interval (ms)": str(
                    patient_data.get("ImageComments", "")
                    .strip("RR ")
                    .replace(" +/- ", "±")
                ),
                "FOV size (rows x cols)": f"{patient_data.get('Rows', '')}x"
                f"{patient_data.get('Columns', '')}",
                "Pixel size (mm)": str(self.data_files.pixel_size(dataset) * 10),
            }
        )
        return output

    def save_all(self):
        """Saves the data to the hdf5 file, if present."""
        if self.strainmap_file is None or self.data_files is None:
            return

        write_hdf5_file(self, self.strainmap_file)

    def save(self, *args):
        """Saves specific attributes to the hdf5 file.

        They must be one of the storable StrainMapData attributes (segments, velocities,
        etc.)
        """
        from .writers import write_data_structure

        if self.strainmap_file is None:
            return

        for key in args:
            if key not in self.stored:
                raise KeyError(f"{key} is not storable.")
            elif key == "timeshift":
                self.strainmap_file.attrs[key] = getattr(self, key)
            else:
                write_data_structure(self.strainmap_file, key, getattr(self, key))

    def delete(self, *args):
        """Deletes the chosen dataset or group from the hdf5 file.

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

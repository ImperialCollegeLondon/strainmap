from pathlib import Path
from typing import Optional, Text, Union

import numpy as np
import xarray as xr

from ..coordinates import Comp
from ..exceptions import NoDICOMDataException
from .readers import DICOMReaderBase, read_folder, read_strainmap_file
from .writers import write_netcdf_file, save_group, save_attribute

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
    def from_folder(cls, data_files: Union[Path, Text]):
        """Creates a new StrainMap data object from a folder containing DICOMs"""
        df = read_folder(data_files)
        if df is None:
            raise NoDICOMDataException(f"'{data_files}' does not contain DICOM data.")
        data = cls(data_files=df)
        return data

    @classmethod
    def from_file(cls, strainmap_file: Union[Path, Text]):
        """Creates a new StrainMap data object from a file."""
        attributes = read_strainmap_file(cls.stored, strainmap_file)
        result = cls()
        result.__dict__.update(attributes)
        return result

    def __init__(
        self, data_files: Optional[DICOMReaderBase] = None, filename: Path = Path()
    ):

        self.data_files = data_files
        self.filename: Path = filename
        self.orientation: str = "CCW"
        self.timeshift: float = TIMESHIFT
        self.sign_reversal: xr.DataArray = xr.DataArray(
            [1, 1, 1],
            dims=["comp"],
            coords={"comp": [Comp.X.name, Comp.Y.name, Comp.Z.name]},
            name="sign_reversal",
        ).astype(np.int16)
        self.segments: xr.DataArray = xr.DataArray(name="segments")
        self.centroid: xr.DataArray = xr.DataArray(name="centroid")
        self.septum: xr.DataArray = xr.DataArray(name="septum")
        self.velocities: xr.DataArray = xr.DataArray(name="velocities")
        self.masks: xr.DataArray = xr.DataArray(name="masks")
        self.markers: xr.DataArray = xr.DataArray(name="markers")
        self.cylindrical: xr.DataArray = xr.DataArray(name="cylindrical")
        self.strain: xr.DataArray = xr.DataArray(name="strain")
        self.strain_markers: xr.DataArray = xr.DataArray(name="strain_markers")
        self.gls: xr.DataArray = xr.DataArray(name="gls")
        self.twist: xr.DataArray = xr.DataArray(name="twist")

    def add_paths(self, data_files: Union[Path, Text] = ""):
        """Adds data paths to the object."""
        self.data_files = read_folder(data_files)
        if self.data_files is None:
            raise NoDICOMDataException(f"'{data_files}' does not contain DICOM data.")

    def add_file(self, strainmap_file: Union[Path, Text]):
        """Adds a new netCDF file to the structure."""
        if not str(strainmap_file).endswith(".nc"):
            return False
        filename = Path(strainmap_file)
        self.filename = filename.parent / filename.name
        self.save_all()
        return True

    def add_data(self, cine: str, **kwargs):
        """Adds new data to the attributes, expanding the 'cine' coordinate.

        After updating the data object, it is saved.

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
                getattr(self, attr).name = attr
            elif cine not in getattr(self, attr).cine:
                setattr(
                    self,
                    attr,
                    xr.concat(
                        [getattr(self, attr), value.expand_dims(cine=[cine]).copy()],
                        dim="cine",
                        fill_value=False,
                    ),
                )
                getattr(self, attr).name = attr
            else:
                getattr(self, attr).loc[{"cine": cine}] = value

        self.save(*kwargs.keys())

    def set_orientation(self, orientation):
        """Sets the angular regions orientation (CW or CCW) and saves the data"""
        self.orientation = orientation
        self.save("orientation")

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
        """Saves the data to the netCDF file, if present."""
        if self.filename == Path(".") or self.data_files is None:
            return

        to_save = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
            and k not in ("filename", "data_files", "strainmap_file")
        }

        write_netcdf_file(self.filename, **to_save, **self.metadata())

    def save(self, *args) -> None:
        """Saves specific object attributes to the hdf5 file.

        DataArrays are saved as groups while anything else is saved as attribute of the
        root group.

        Args:
            - args: Names of instance attributes to save.

        Return:
            None
        """
        if self.filename == Path("."):
            return

        for key in args:
            value = getattr(self, key)
            if isinstance(value, xr.DataArray):
                save_group(self.filename, value, key, value.dtype == float)
            else:
                save_attribute(self.filename, **{key: value})

    def __eq__(self, other) -> bool:
        """Compares two StrainMapData objects.

        The "strainmap_file" attribute is ignored as it might have different values."""
        equal = self.sign_reversal == other.sign_reversal
        keys = set(self.__dict__.keys())
        for k in keys:
            equal = equal and compare_dicts(getattr(self, k), getattr(other, k))
        return equal

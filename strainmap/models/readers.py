from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence, Text, Tuple, Union

import h5py
import numpy as np
import pydicom
import xarray as xr
from natsort import natsorted

from ..coordinates import Comp


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    Function taken from https://stackoverflow.com/a/312464/3778792
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parallel_spirals(header):
    """Checks if ParallelSpirals is in the tSequenceFileName."""
    try:
        tSequenceFileName = re.search('tSequenceFileName\t = \t""(.*)""', header)
        if tSequenceFileName is None:
            return False
        return (
            re.search(r"(.*)Parallel(.?)Spirals(.?)", tSequenceFileName[1]) is not None
        )
    except TypeError:
        return False


def velocity_sensitivity(header) -> xr.DataArray:
    """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
    z = float(re.search(r"asElm\[0\].nVelocity\t = \t(.*)", header)[1])
    r = float(re.search(r"asElm\[1\].nVelocity\t = \t(.*)", header)[1])
    theta = float(re.search(r"asElm\[2\].nVelocity\t = \t(.*)", header)[1])

    return xr.DataArray(
        np.array((r, theta, z)) * 2,
        dims=["comp"],
        coords={"comp": [Comp.RAD.name, Comp.CIRC.name, Comp.LONG.name]},
    )


def phase_encoding(filename) -> Tuple[bool, xr.DataArray]:
    """Indicates if X and Y Phases should be swapped and the velocity sign factors."""
    ds = pydicom.dcmread(filename)
    swap = ds.InPlanePhaseEncodingDirection == "ROW"
    signs = xr.DataArray(
        [-1, -1, -1] if swap else [1, 1, 1],
        dims=["comp"],
        coords={"comp": [Comp.RAD.name, Comp.CIRC.name, Comp.LONG.name]},
    )
    return swap, signs


def read_images(filenames: Sequence[str]) -> np.ndarray:
    """Returns the images for a given series and var."""
    return np.array([pydicom.dcmread(f).pixel_array for f in filenames])


def read_strainmap_file(filename: Union[Path, Text], stored: Tuple = ()) -> dict:
    """Reads a StrainMap file with existing information on previous segmentations."""
    fn = Path(filename)
    if fn.suffix == ".h5":
        return read_h5_file(stored, fn)
    elif fn.suffix == ".nc":
        return read_netcdf_file(fn)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_h5_file(stored: Tuple, filename: Union[Path, Text]) -> dict:
    """Reads a HDF5 file."""
    sm_file = h5py.File(filename, "a")
    attributes = dict(strainmap_file=filename)

    to_rename = {}

    def search_to_rename(name: str):
        if " - " in name:
            new_name = name.split(" - ")[0]
            to_rename[name] = new_name

    sm_file.visit(search_to_rename)
    for k, v in to_rename.items():
        sm_file.move(k, v)

    for s in stored:
        if s not in sm_file and s not in sm_file.attrs:
            continue

        if s == "sign_reversal":
            attributes[s] = tuple(sm_file[s][...])
        elif s == "orientation":
            attributes[s] = str(sm_file[s][...])
        elif s == "timeshift":
            # TODO Simplify in the final version. Current design "heals" existing files
            if s in sm_file.attrs:
                attributes[s] = sm_file.attrs[s]
            elif s in sm_file:
                del sm_file[s]
                continue
        elif "files" in s:
            base_dir = paths_from_hdf5(defaultdict(dict), filename, sm_file[s])
            if base_dir is None:
                attributes[s] = ()
            else:
                attributes[s] = read_folder(base_dir)
        else:
            attributes[s] = defaultdict(dict)
            read_data_structure(attributes[s], sm_file[s])

    return attributes


def read_data_structure(g, structure):
    """Recursively populates the StrainData object with the contents of the hdf5
    file."""
    for n, struct in structure.items():
        if isinstance(struct, h5py.Group):
            if len(struct.keys()) == 0:
                del structure[n]
                continue
            read_data_structure(g[n], struct)
        else:
            g[n] = struct[...]


def from_relative_paths(master: str, paths: List[bytes]) -> list:
    """Transform a list of relative paths to a given master to absolute paths."""
    import sys

    if sys.platform == "win32":
        return []

    return [
        str((Path(master).parent / PurePath(PurePosixPath(p.decode()))).resolve())
        for p in paths
    ]


def paths_from_hdf5(g, master, structure):
    """Populates the StrainData object with the paths contained in the hdf5 file."""
    base_dir = None
    for n, struct in structure.items():
        if isinstance(struct, h5py.Group):
            base_dir = paths_from_hdf5(g[n], master, struct)
        else:
            filenames = from_relative_paths(master, struct[...])
            if len(filenames) > 0 and all(map(os.path.isfile, filenames)):
                g[n] = filenames
                base_dir = Path(filenames[0]).parent

    return base_dir


class DICOMReaderBase(ABC):
    """Base class for all the DICOM file readers."""

    @property
    @abstractmethod
    def vars(self) -> dict:
        """Equivalence between general var names and actual ones.

        eg. [Mag, PhaseX, PhaseY, PhaseZ] -> [MagZ, PhaseX, PhaseY, PhaseZ]
        """

    @staticmethod
    @abstractmethod
    def belongs(path: Union[Path, Text]) -> bool:
        """Indicates if the input file is compatible with this reader_class.

        This could be by analysing the filename itself or some specific content within
        the file.
        """

    @classmethod
    @abstractmethod
    def factory(cls, path: Union[Path, Text]):
        """Reads the dicom files in the directory indicated in path."""

    def __init__(self, files: xr.DataArray):
        self.files = files

    @property
    def frames(self) -> int:
        """Number of frames per dataset."""
        return self.files.sizes["frame"]

    @property
    def datasets(self) -> List[str]:
        """List of datasets available in the files."""
        return self.files.cine.values.tolist()

    @property
    def is_avail(self) -> Union[str, bool]:
        """Provides the first filename or False if there are no files."""
        return (
            False if self.files[0, 0, 0].item() is None else self.files[0, 0, 0].item()
        )

    @property
    @abstractmethod
    def sensitivity(self) -> np.ndarray:
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""

    def phase_encoding(self, dataset: str) -> Tuple[bool, xr.DataArray]:
        """Indicates if X-Y Phases should be swapped and the velocity sign factors."""
        return phase_encoding(self.files.sel(cine=dataset)[0, 0].item())

    def tags(self, dataset: str, var: Optional[str] = None) -> dict:
        """Dictionary with the tags available in the DICOM files."""
        data_dict = dict()

        if var is not None:
            filename = self.files.sel(cine=dataset, raw_comp=self.vars[var])[0].item()
        else:
            filename = self.files.sel(cine=dataset)[0, 0].item()

        if filename is not None:
            data = pydicom.dcmread(filename)

            for i, d in enumerate(data.dir()):
                data_dict[d] = getattr(data, d)

        return data_dict

    def tag(self, dataset: str, tag: str) -> Any:
        """Return the requested tag from the first file of chosen dataset."""
        filename = self.files.sel(cine=dataset)[0, 0].item()

        if filename is not None:
            return getattr(pydicom.dcmread(filename), tag)

    @abstractmethod
    def cine_loc(self, dataset: str) -> float:
        """Returns the cine location in cm from the isocentre."""

    @abstractmethod
    def pixel_size(self, dataset: str) -> float:
        """Returns the pixel size in cm."""

    @abstractmethod
    def time_interval(self, dataset: str) -> float:
        """Returns the frame time interval in seconds."""

    @abstractmethod
    def mag(self, dataset: str) -> np.ndarray:
        """Provides the magnitude data corresponding to the chosen dataset.

        The expected shape of the array is [frames, xpoints, ypoints].
        """

    @abstractmethod
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset.

        The expected shape of the array is [3, frames, xpoints, ypoints]. The components
        should follow the order PhaseX -> PhaseY -> PhaseZ.
        """

    @abstractmethod
    def images(self, dataset: str) -> xr.DataArray:
        """Returns one specific component of all the image data."""


DICOM_READERS = []
"""List of available DICOM readers."""


def register_dicom_reader(reader_class):
    """Registers the reader_class in the list of available readers."""
    if issubclass(reader_class, DICOMReaderBase) and set(reader_class.vars.keys()) == {
        "MAG",
        "X",
        "Y",
        "Z",
    }:
        DICOM_READERS.append(reader_class)
    return reader_class


def read_folder(path: Union[Path, Text, None]) -> Optional[DICOMReaderBase]:
    """Find a reader appropriate to read the contents of the given folder."""
    for r in DICOM_READERS:
        if r.belongs(path):
            return r.factory(path)
    return None


@register_dicom_reader
class DICOM(DICOMReaderBase):

    variables = ["MagAvg", "PhaseZ", "PhaseX", "PhaseY", "RefMag"]

    vars = {"MAG": "MagAvg", "Z": "PhaseZ", "X": "PhaseX", "Y": "PhaseY"}

    @staticmethod
    def belongs(path: Union[Path, Text]) -> bool:
        """Indicates if the input folder is compatible with this reader_class."""
        path = str(Path(path) / "*.dcm")
        filenames = sorted(glob.glob(path))
        return (
            len(filenames) > 0
            and all([len(Path(p).stem.split(".")) == 13 for p in filenames])
            and "ImageComments" in pydicom.dcmread(filenames[0])
        )

    @classmethod
    def factory(cls, path: Union[Path, Text]) -> DICOM:
        """Reads in the legacy Dicom data files."""
        pattern = str(Path(path) / "*.*.*.*.1.*.*.*.*.*.*.*.*.dcm")
        filenames = chunks(natsorted(glob.glob(pattern)), len(cls.variables))

        cine = []
        data = []
        for f in filenames:
            if len(f) != len(cls.variables):
                continue

            ds = pydicom.dcmread(f[0])
            header = ds[("0021", "1019")].value.decode()
            if not parallel_spirals(header):
                continue

            cine.append(f"{ds.SeriesNumber} {ds.SeriesDescription}")
            data.append([])

            for i, var in enumerate(cls.variables):
                num = Path(f[i]).stem.split(".")[3]
                data[-1].append(
                    natsorted(
                        glob.glob(
                            str(Path(path) / f"*.*.*.{num}.*.*.*.*.*.*.*.*.*.dcm")
                        )
                    )
                )

        return cls(
            xr.DataArray(
                data,
                dims=["cine", "raw_comp", "frame"],
                coords={"cine": cine, "raw_comp": cls.variables},
            )
        )

    @property
    def sensitivity(self) -> xr.DataArray:
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
        ds = pydicom.dcmread(self.is_avail)
        header = ds[("0021", "1019")].value.decode()
        return velocity_sensitivity(header)

    def cine_loc(self, dataset: str) -> float:
        """Returns the cine location in cm from the isocentre."""
        return float(self.tags(dataset)["SliceLocation"]) / 10.0

    def pixel_size(self, dataset: str) -> float:
        """Returns the pixel size in cm."""
        return float(self.tags(dataset)["PixelSpacing"][0]) / 10.0

    def time_interval(self, dataset: str) -> float:
        """Returns the frame time interval in seconds."""
        return float(self.tags(dataset)["ImageComments"].split(" ")[1]) / 1000.0 / 50

    def mag(self, dataset: str) -> np.ndarray:
        """Provides the Magnitude data corresponding to the chosen dataset."""
        return read_images(self.files.sel(cine=dataset, raw_comp="MagAvg").values)

    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset."""
        phasex = read_images(self.files.sel(cine=dataset, raw_comp="PhaseX").values)
        phasey = read_images(self.files.sel(cine=dataset, raw_comp="PhaseY").values)
        phasez = read_images(self.files.sel(cine=dataset, raw_comp="PhaseZ").values)

        return np.stack((phasex, phasey, phasez))

    @lru_cache(1)
    def images(self, dataset: str) -> xr.DataArray:
        """Provides the Phase data corresponding to the chosen dataset."""
        mag = read_images(self.files.sel(cine=dataset, raw_comp="MagAvg").values)
        phasex = read_images(self.files.sel(cine=dataset, raw_comp="PhaseX").values)
        phasey = read_images(self.files.sel(cine=dataset, raw_comp="PhaseY").values)
        phasez = read_images(self.files.sel(cine=dataset, raw_comp="PhaseZ").values)

        return xr.DataArray(
            np.stack((mag, phasex, phasey, phasez)),
            dims=["comp", "frame", "row", "col"],
            coords={
                "comp": [Comp.MAG.name, Comp.X.name, Comp.Y.name, Comp.Z.name],
                "frame": np.arange(0, mag.shape[0]),
                "cine": dataset,
                "row": np.arange(0, mag.shape[1]),
                "col": np.arange(0, mag.shape[2]),
            },
        )


def read_netcdf_file(filename: Path) -> Dict:
    """Read a netCDF file and returns a dictionary with its contents.

    The file is assumed to have been created with StrainMap (write_netcdf_file) and have
    all DataArrays as groups and all non-DataArray information as attributes of the
    root group.

    Args:
        - filename: The name of the file to read the data from.

    Return:
        A dictionary with contents of the file.
    """
    if filename.suffix != ".nc":
        raise ValueError(
            f"'{filename.suffix}' is an invalid extension for a netCDF file. It must "
            "be '.nc'."
        )

    result = dict(filename=filename)
    with h5py.File(filename, "r") as f:
        for k, v in f.attrs.items():
            if k.isidentifier():
                result[k] = v
        names = tuple(f.keys())

    for n in names:
        with xr.open_dataarray(filename, group=n) as da:
            result[n] = da.load()

    return result

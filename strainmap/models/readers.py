import glob
import re
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PurePath
from typing import (
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Text,
    Tuple,
    Union,
    NoReturn,
    Type,
)
from abc import ABC, abstractmethod
from functools import lru_cache
from collections import defaultdict

import pydicom
from nibabel.nicom import csareader as csar
import numpy as np
import h5py


VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


def read_dicom_directory_tree(path: Union[Path, Text]) -> Mapping:
    """Creates a dictionary with the available series and associated
    filenames."""

    path = str(Path(path) / "*01.dcm")
    filenames = sorted(glob.glob(path))

    data_files: OrderedDict = OrderedDict()
    var_idx: Dict = {}
    for f in filenames:
        ds = pydicom.dcmread(f)

        if not parallel_spirals(ds):
            continue

        name = ds.SeriesDescription
        if name not in data_files.keys():
            data_files[name] = OrderedDict()
            var_idx = {}
            for var in VAR_OFFSET:
                data_files[name][var] = []
                var_idx[int(Path(f).name[3:5]) + VAR_OFFSET[var]] = var

        data_files[name][var_idx[int(Path(f).name[3:5])]] = sorted(
            glob.glob(f.replace("01.dcm", "*.dcm"))
        )

    return data_files


def parallel_spirals(dicom_data):
    """Checks if ParallelSpirals is in the tSequenceFileName."""
    csa = csar.get_csa_header(dicom_data, "series")
    try:
        ascii_header = csa["tags"]["MrPhoenixProtocol"]["items"][0]
        tSequenceFileName = re.search('tSequenceFileName\t = \t""(.*)""', ascii_header)
        if tSequenceFileName is None:
            return False
        return (
            re.search(r"(.*)Parallel(.?)Spirals(.?)", tSequenceFileName[1]) is not None
        )
    except TypeError:
        return False


def velocity_sensitivity(filename):
    """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
    dicom_data = pydicom.dcmread(filename)
    csa = csar.get_csa_header(dicom_data, "series")

    ascii_header = csa["tags"]["MrPhoenixProtocol"]["items"][0]
    z = float(re.search(r"asElm\[0\].nVelocity\t = \t(.*)", ascii_header)[1])
    r = float(re.search(r"asElm\[1\].nVelocity\t = \t(.*)", ascii_header)[1])
    theta = float(re.search(r"asElm\[2\].nVelocity\t = \t(.*)", ascii_header)[1])

    return np.array((z, r, theta)) * 2


def image_orientation(filename):
    """Indicates if X and Y Phases should be swapped and the velocity sign factors."""
    ds = pydicom.dcmread(filename)
    swap = ds.InPlanePhaseEncodingDirection == "ROW"
    signs = np.array([1, -1, 1]) * (-1) ** swap
    return swap, signs


def read_dicom_file_tags(
    origin: Union[Mapping, Path, Text],
    series: Optional[Text] = None,
    variable: Optional[Text] = None,
    timestep: Optional[int] = None,
) -> Mapping:
    """Returns a dictionary with the tags and values available in a DICOM
    file."""
    if isinstance(origin, Mapping):
        if len(origin) == 0:
            return {}

        assert series in origin
        assert variable in origin[series]
        assert isinstance(timestep, int) and timestep < len(origin[series][variable])
        filename = origin[series][variable][timestep]
    else:
        filename = origin

    data = pydicom.dcmread(filename)

    data_dict: OrderedDict = OrderedDict()
    for i, d in enumerate(data.dir()):
        data_dict[d] = getattr(data, d)

    return data_dict


def read_images(origin: Mapping, series: Text, variable: Text) -> List:
    """Returns the images for a given series and var."""
    if series in origin and variable in origin[series]:
        return [pydicom.dcmread(f).pixel_array for f in origin[series][variable]]
    else:
        return []


def read_all_images(origin: Mapping) -> Mapping:
    """Returns all the images organised by series and variables."""
    return {
        series: {
            variable: [pydicom.dcmread(f).pixel_array for f in images]
            for variable, images in variables.items()
        }
        for series, variables in origin.items()
    }


@dataclass
class ImageTimeSeries:
    """Aggregates dicom images into two numpy arrays.

    Each arrays has the format (vector component, time, horizontal, vertical).
    """

    magnitude: np.ndarray
    phase: np.ndarray

    component_axis: ClassVar[int] = 0
    """Axis with x, y, z components."""
    time_axis: ClassVar[int] = 1
    """Axis representing time."""
    image_axes: ClassVar[Tuple[int, int]] = (2, 3)
    """Axis representing images"""

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            return list(self[i] for i in index)
        elif isinstance(index, slice):
            return self[range(*index.indices(2))]
        elif index == 0:
            return self.magnitude
        elif index == 1:
            return self.phase
        else:
            raise IndexError(f"Invalid index {index}")

    def __len__(self):
        return 2

    def __iter__(self):
        return iter((self.magnitude, self.phase))


def images_to_numpy(data: Mapping) -> Mapping[Text, ImageTimeSeries]:
    """Aggregates dicom images into numpy arrays."""

    def to_numpy(pattern: Text, data: Mapping) -> np.ndarray:
        from numpy import stack, argsort

        result = stack(
            tuple(
                stack(data[key]) for key in map(lambda x: pattern + x, ("X", "Y", "Z"))
            )
        )
        return result.transpose(
            argsort(
                (
                    ImageTimeSeries.component_axis,
                    ImageTimeSeries.time_axis,
                    *ImageTimeSeries.image_axes,
                )
            )
        )

    return {
        k: ImageTimeSeries(to_numpy("Mag", v), to_numpy("Phase", v))
        for k, v in data.items()
    }


def read_strainmap_file(data, filename: Union[Path, Text]):
    """Reads a StrainMap file with existing information on previous segmentations."""
    if str(filename).endswith(".h5"):
        return read_h5_file(data, filename)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_h5_file(data, filename: Union[Path, Text]):
    """Reads a HDF5 file."""
    sm_file = h5py.File(filename, "a")
    data.strainmap_file = sm_file

    for s in data.stored:
        if s == "sign_reversal":
            data.sign_reversal = tuple(sm_file[s][...])
        elif "files" in s and s in sm_file:
            base_dir = paths_from_hdf5(defaultdict(dict), filename, sm_file[s])
            if base_dir is None:
                setattr(data, s, ())
            else:
                setattr(data, s, read_folder(base_dir))
        elif s in sm_file:
            read_data_structure(getattr(data, s), sm_file[s])

    return data


def read_data_structure(g, structure):
    """Recursively populates the StrainData object with the contents of the hdf5 file.
    """
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
    return [
        str((Path(master).parent / PurePath(PurePosixPath(p.decode()))).resolve())
        for p in paths
    ]


def paths_from_hdf5(g, master, structure):
    """Populates the StrainData object with the paths contained in the hdf5 file.
    """
    base_dir = None
    for n, struct in structure.items():
        if isinstance(struct, h5py.Group):
            base_dir = paths_from_hdf5(g[n], master, struct)
        else:
            filenames = from_relative_paths(master, struct[...])
            if all(map(os.path.isfile, filenames)):
                g[n] = filenames
                base_dir = Path(filenames[0]).parent

    return base_dir


class DICOMReaderBase(ABC):
    """Base class for all the DICOM file readers"""

    @property
    @classmethod
    @abstractmethod
    def vars(cls) -> dict:
        """Equivalence between general var names and actual ones.

        eg. [Mag, PhaseX, PhaseY, PhaseZ] -> [MagZ, PhaseX, PhaseY, PhaseZ]"""

    @staticmethod
    @abstractmethod
    def belongs(path: Union[Path, Text]) -> bool:
        """Indicates if the input file is compatible with this reader_class.

        This could be by analysing the filename itself or some specific content
        within the file."""

    @classmethod
    @abstractmethod
    def factory(cls, path: Union[Path, Text]):
        """Reads the dicom files in the directory indicated in path."""

    def __init__(self, files: OrderedDict):
        self.files = files

    @property
    def datasets(self) -> list:
        """List of datasets available in the files."""
        return list(self.files.keys())

    @property
    def is_avail(self) -> Union[str, bool]:
        """Provides the first filename or False if there are no files."""
        var = list(self.files[self.datasets[0]].values())
        return var[0][0] if len(var) > 0 and len(var[0]) > 0 else False

    @property
    def sensitivity(self):
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
        return velocity_sensitivity(self.is_avail)

    @property
    def orientation(self):
        """Indicates if X-Y Phases should be swapped and the velocity sign factors."""
        return image_orientation(self.is_avail)

    def tags(self, dataset: str, var: Optional[str] = None) -> dict:
        """Dictionary with the tags available in the DICOM files."""
        data_dict = dict()

        if var is not None:
            var = self.files[dataset].get(self.vars[var], [])
        else:
            var = list(self.files[dataset].values())
            var = var[0] if len(var) > 0 else []

        if len(var) > 0:
            data = pydicom.dcmread(var[0])

            for i, d in enumerate(data.dir()):
                data_dict[d] = getattr(data, d)

        return data_dict

    @abstractmethod
    def slice_loc(self, dataset: str) -> float:
        """Returns the slice location in cm from the isocentre."""

    @abstractmethod
    def pixel_size(self, dataset: str) -> float:
        """Returns the pixel size in cm."""

    @abstractmethod
    def time_interval(self, dataset: str) -> float:
        """Returns the frame time interval in seconds."""

    @lru_cache(1)
    @abstractmethod
    def mag(self, dataset: str) -> np.ndarray:
        """Provides the magnitude data corresponding to the chosen dataset.

        The expected shape of the array is [frames, xpoints, ypoints]."""

    @lru_cache(1)
    @abstractmethod
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset.

        The expected shape of the array is [3, frames, xpoints, ypoints]. The components
        should follow the order PhaseX -> PhaseY -> PhaseZ."""

    def images(self, dataset: str, var: str) -> np.ndarray:
        """Returns one specific component of all the image data."""
        nvar = list(self.vars.keys()).index(var)
        return self.mag(dataset) if nvar == 0 else self.phase(dataset)[nvar - 1]


DICOM_READERS = []
"""List of available DICOM readers."""


def register_dicom_reader(reader_class):
    """Registers the reader_class in the list of available readers."""
    if issubclass(reader_class, DICOMReaderBase) and set(reader_class.vars.keys()) == {
        "Mag",
        "PhaseX",
        "PhaseY",
        "PhaseZ",
    }:
        DICOM_READERS.append(reader_class)
    return reader_class


def read_folder(path: Union[Path, Text, None]) -> Optional[Type[DICOMReaderBase]]:
    """Find a reader appropriate to read the contents of the given folder."""
    for r in DICOM_READERS:
        if r.belongs(path):
            return r.factory(path)


@register_dicom_reader
class LegacyDICOM(DICOMReaderBase):

    offset = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}

    vars = {"Mag": "MagZ", "PhaseZ": "PhaseZ", "PhaseX": "PhaseX", "PhaseY": "PhaseY"}

    @staticmethod
    def belongs(path: Union[Path, Text]) -> bool:
        """Indicates if the input folder is compatible with this reader_class."""
        path = str(Path(path) / "*.dcm")
        filenames = sorted(glob.glob(path))
        return len(filenames) > 0 and all(
            [len(Path(p).stem) == 11 and Path(p).stem[:2] == "MR" for p in filenames]
        )

    @classmethod
    def factory(cls, path: Union[Path, Text]):
        """Reads in the legacy Dicom data files."""
        path = str(Path(path) / "*00.dcm")
        filenames = sorted(glob.glob(path))

        data_files: OrderedDict = OrderedDict()
        var_idx: Dict = {}
        for f in filenames:
            ds = pydicom.dcmread(f)

            if not parallel_spirals(ds):
                continue

            name = ds.SeriesDescription
            if name not in data_files.keys():
                data_files[name] = OrderedDict()
                var_idx = {}
                for var in cls.offset:
                    data_files[name][var] = []
                    var_idx[int(Path(f).name[3:5]) + cls.offset[var]] = var

            data_files[name][var_idx[int(Path(f).name[3:5])]] = sorted(
                glob.glob(f.replace("00.dcm", "*.dcm"))
            )

        return cls(data_files)

    def slice_loc(self, dataset: str) -> NoReturn:
        """Returns the slice location in cm from the isocentre."""
        raise AttributeError("LegacyDICOM has no slice location defined.")

    def pixel_size(self, dataset: str) -> NoReturn:
        """Returns the pixel size in cm."""
        raise AttributeError("LegacyDICOM has no pixel size defined.")

    def time_interval(self, dataset: str) -> NoReturn:
        """Returns the frame time interval in seconds."""
        raise AttributeError("LegacyDICOM has no time interval defined.")

    @lru_cache(1)
    def mag(self, dataset: str) -> np.ndarray:
        """Provides the Magnitude data corresponding to the chosen dataset."""
        magx = np.array(read_images(self.files, dataset, "MagX"))
        magy = np.array(read_images(self.files, dataset, "MagY"))
        magz = np.array(read_images(self.files, dataset, "MagZ"))

        return (magx + magy + magz) / 3

    @lru_cache(1)
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset."""
        phasex = np.array(read_images(self.files, dataset, "PhaseX"))
        phasey = np.array(read_images(self.files, dataset, "PhaseY"))
        phasez = np.array(read_images(self.files, dataset, "PhaseZ"))

        return np.stack((phasex, phasey, phasez))

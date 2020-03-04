import glob
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePath, PurePosixPath
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Text, Tuple, Union

import h5py
import numpy as np
import pydicom
from natsort import natsorted

VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


def read_dicom_directory_tree(path: Union[Path, Text]) -> Mapping:
    """Creates a dictionary with the available series and associated filenames."""
    from nibabel.nicom import csareader as csar

    path = str(Path(path) / "*01.dcm")
    filenames = sorted(glob.glob(path))

    data_files: OrderedDict = OrderedDict()
    var_idx: Dict = {}
    for f in filenames:
        ds = pydicom.dcmread(f)
        csa = csar.get_csa_header(ds, "series")
        header = csa.get("tags", {}).get("MrPhoenixProtocol", {}).get("items", [])
        if len(header) == 0 or not parallel_spirals(header[0]):
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


def velocity_sensitivity(header) -> np.ndarray:
    """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
    z = float(re.search(r"asElm\[0\].nVelocity\t = \t(.*)", header)[1])
    r = float(re.search(r"asElm\[1\].nVelocity\t = \t(.*)", header)[1])
    theta = float(re.search(r"asElm\[2\].nVelocity\t = \t(.*)", header)[1])

    return np.array((z, r, theta)) * 2


def image_orientation(filename) -> tuple:
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
    """Returns a dictionary with the tags and values available in a DICOM file."""
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


def read_strainmap_file(stored: Tuple, filename: Union[Path, Text]) -> dict:
    """Reads a StrainMap file with existing information on previous segmentations."""
    if str(filename).endswith(".h5"):
        return read_h5_file(stored, filename)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_h5_file(stored: Tuple, filename: Union[Path, Text]) -> dict:
    """Reads a HDF5 file."""
    sm_file = h5py.File(filename, "a")
    attributes = dict(strainmap_file=sm_file)

    for s in stored:
        if s == "sign_reversal":
            attributes[s] = tuple(sm_file[s][...])
        elif "files" in s and s in sm_file:
            base_dir = paths_from_hdf5(defaultdict(dict), filename, sm_file[s])
            if base_dir is None:
                attributes[s] = ()
            else:
                attributes[s] = read_folder(base_dir)
        elif s in sm_file:
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
            if all(map(os.path.isfile, filenames)):
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

    def __init__(self, files: dict):
        self.files = files

    @property
    def datasets(self) -> List[str]:
        """List of datasets available in the files."""
        return list(self.files.keys())

    @property
    def is_avail(self) -> Union[str, bool]:
        """Provides the first filename or False if there are no files."""
        var = list(self.files[self.datasets[0]].values())
        return var[0][0] if len(var) > 0 and len(var[0]) > 0 else False

    @property
    @abstractmethod
    def sensitivity(self) -> np.ndarray:
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""

    @property
    def orientation(self) -> tuple:
        """Indicates if X-Y Phases should be swapped and the velocity sign factors."""
        return image_orientation(self.is_avail)

    def tags(self, dataset: str, var: Optional[str] = None) -> dict:
        """Dictionary with the tags available in the DICOM files."""
        data_dict = dict()

        if var is not None:
            files = self.files[dataset].get(self.vars[var], [])
        else:
            files = list(self.files[dataset].values())
            files = files[0] if len(files) > 0 else []

        if len(files) > 0:
            data = pydicom.dcmread(files[0])

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

        The expected shape of the array is [frames, xpoints, ypoints].
        """

    @lru_cache(1)
    @abstractmethod
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset.

        The expected shape of the array is [3, frames, xpoints, ypoints]. The components
        should follow the order PhaseX -> PhaseY -> PhaseZ.
        """

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


def read_folder(path: Union[Path, Text, None]) -> Optional[DICOMReaderBase]:
    """Find a reader appropriate to read the contents of the given folder."""
    for r in DICOM_READERS:
        if r.belongs(path):
            return r.factory(path)
    return None


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
        from nibabel.nicom import csareader as csar

        path = str(Path(path) / "*00.dcm")
        filenames = natsorted(glob.glob(path))

        data_files: dict = dict()
        var_idx: Dict = {}
        for f in filenames:
            ds = pydicom.dcmread(f)

            csa = csar.get_csa_header(ds, "series")
            header = csa.get("tags", {}).get("MrPhoenixProtocol", {}).get("items", [])
            if len(header) == 0 or not parallel_spirals(header[0]):
                continue

            name = ds.SeriesDescription
            if name not in data_files.keys():
                data_files[name] = dict()
                var_idx = dict()
                for var in cls.offset:
                    data_files[name][var] = []
                    var_idx[int(Path(f).name[3:5]) + cls.offset[var]] = var

            data_files[name][var_idx[int(Path(f).name[3:5])]] = sorted(
                glob.glob(f.replace("00.dcm", "*.dcm"))
            )

        return cls(data_files)

    @property
    def sensitivity(self) -> np.ndarray:
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
        from nibabel.nicom import csareader as csar

        ds = pydicom.dcmread(self.is_avail)
        csa = csar.get_csa_header(ds, "series")
        header = csa.get("tags", {}).get("MrPhoenixProtocol", {}).get("items", [])[0]
        return velocity_sensitivity(header)

    def slice_loc(self, dataset: str) -> None:
        """Returns the slice location in cm from the isocentre."""
        raise AttributeError("LegacyDICOM has no slice location defined.")

    def pixel_size(self, dataset: str) -> None:
        """Returns the pixel size in cm."""
        raise AttributeError("LegacyDICOM has no pixel size defined.")

    def time_interval(self, dataset: str) -> None:
        """Returns the frame time interval in seconds."""
        raise AttributeError("LegacyDICOM has no time interval defined.")

    @lru_cache(1)
    def mag(self, dataset: str) -> np.ndarray:
        """Provides the Magnitude data corresponding to the chosen dataset."""
        magx = np.array(read_images(self.files, dataset, "MagX"))
        magy = np.array(read_images(self.files, dataset, "MagY"))
        magz = np.array(read_images(self.files, dataset, "MagZ"))

        return (magx + magy + magz) // 3

    @lru_cache(1)
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset."""
        phasex = np.array(read_images(self.files, dataset, "PhaseX"))
        phasey = np.array(read_images(self.files, dataset, "PhaseY"))
        phasez = np.array(read_images(self.files, dataset, "PhaseZ"))

        return np.stack((phasex, phasey, phasez))


@register_dicom_reader
class DICOM(DICOMReaderBase):

    variables = ["MagAvg", "PhaseZ", "PhaseX", "PhaseY", "RefMag"]

    vars = {"Mag": "MagAvg", "PhaseZ": "PhaseZ", "PhaseX": "PhaseX", "PhaseY": "PhaseY"}

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
    def factory(cls, path: Union[Path, Text]):
        """Reads in the legacy Dicom data files."""
        pattern = str(Path(path) / "*.*.*.*.1.*.*.*.*.*.*.*.*.dcm")
        filenames = chunks(natsorted(glob.glob(pattern)), len(cls.variables))

        data_files: dict = dict()
        for f in filenames:
            if len(f) != len(cls.variables):
                continue

            ds = pydicom.dcmread(f[0])
            header = ds[("0021", "1019")].value.decode()
            if not parallel_spirals(header):
                continue

            dataset_name = f"{ds.SeriesNumber} {ds.SeriesDescription}"
            data_files[dataset_name] = dict()

            for i, var in enumerate(cls.variables):
                num = Path(f[i]).stem.split(".")[3]
                data_files[dataset_name][var] = natsorted(
                    glob.glob(str(Path(path) / f"*.*.*.{num}.*.*.*.*.*.*.*.*.*.dcm"))
                )

        return cls(data_files)

    @property
    def sensitivity(self) -> np.ndarray:
        """Obtains the in-plane and out of plane velocity sensitivity (scale)."""
        ds = pydicom.dcmread(self.is_avail)
        header = ds[("0021", "1019")].value.decode()
        return velocity_sensitivity(header)

    def slice_loc(self, dataset: str) -> float:
        """Returns the slice location in cm from the isocentre."""
        return float(self.tags(dataset)["SliceLocation"]) / 10.0

    def pixel_size(self, dataset: str) -> float:
        """Returns the pixel size in cm."""
        return float(self.tags(dataset)["PixelSpacing"][0]) / 10.0

    def time_interval(self, dataset: str) -> float:
        """Returns the frame time interval in seconds."""
        return float(self.tags(dataset)["ImageComments"].split(" ")[1]) / 1000.0

    @lru_cache(1)
    def mag(self, dataset: str) -> np.ndarray:
        """Provides the Magnitude data corresponding to the chosen dataset."""
        return np.array(read_images(self.files, dataset, "MagAvg"))

    @lru_cache(1)
    def phase(self, dataset: str) -> np.ndarray:
        """Provides the Phase data corresponding to the chosen dataset."""
        phasex = np.array(read_images(self.files, dataset, "PhaseX"))
        phasey = np.array(read_images(self.files, dataset, "PhaseY"))
        phasez = np.array(read_images(self.files, dataset, "PhaseZ"))

        return np.stack((phasex, phasey, phasez))


@register_dicom_reader
class DICOMNoTimeInterval(DICOM):
    """TODO: Remove when final release. This subclass is necessary just because in the
        initial new scanners the time interval was missing from the DICOM."""

    @staticmethod
    def belongs(path: Union[Path, Text]) -> bool:
        """Indicates if the input folder is compatible with this reader_class."""
        path = str(Path(path) / "*.dcm")
        filenames = sorted(glob.glob(path))
        return (
            len(filenames) > 0
            and all([len(Path(p).stem.split(".")) == 13 for p in filenames])
            and "ImageComments" not in pydicom.dcmread(filenames[0])
        )

    def time_interval(self, dataset: str) -> float:
        """Returns the frame time interval in seconds."""
        if isinstance(self.is_avail, str):
            return float(Path(self.is_avail).parent.name.split("RR")[-1]) / 1000.0
        else:
            raise AttributeError

import glob
import re
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PurePath
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Text, Tuple, Union

import pydicom
from nibabel.nicom import csareader as csar
import numpy as np
import h5py


VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


def read_dicom_directory_tree(path: Union[Path, Text]) -> Mapping:
    """Creates a dictionary with the available series and associated
    filenames."""

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
            for var in VAR_OFFSET:
                data_files[name][var] = []
                var_idx[int(Path(f).name[3:5]) + VAR_OFFSET[var]] = var

        data_files[name][var_idx[int(Path(f).name[3:5])]] = sorted(
            glob.glob(f.replace("00.dcm", "*.dcm"))
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
        return True if "ParallelSpirals" in tSequenceFileName[1] else False
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
    """Returns the images for a given series and variable."""
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


def read_strainmap_file(filename: Union[Path, Text]):
    """Reads a StrainMap file with existing information on previous segmentations."""
    if str(filename).endswith(".h5"):
        return read_h5_file(filename)
    elif str(filename).endswith(".m"):
        return read_matlab_file(filename)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_matlab_file(filename: Union[Path, Text]):
    """Reads a Matlab file."""
    raise NotImplementedError


def read_h5_file(filename: Union[Path, Text]):
    """Reads a HDF5 file."""
    from .strainmap_data_model import factory

    sm_file = h5py.File(filename, "a")

    data = factory()

    for s in data.__dict__.keys():
        if s == "strainmap_file":
            data.strainmap_file = sm_file
        elif s == "sign_reversal":
            data.sign_reversal = tuple(sm_file[s][...])
        elif "files" in s:
            paths_from_hdf5(getattr(data, s), filename, sm_file[s])
        else:
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
    for n, struct in structure.items():
        if isinstance(struct, h5py.Group):
            paths_from_hdf5(g[n], master, struct)
        else:
            filenames = from_relative_paths(master, struct[...])
            g[n] = filenames if all(map(os.path.isfile, filenames)) else []

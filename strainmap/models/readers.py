import glob
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Text, Union

import pydicom
from nibabel.nicom import csareader as csar

VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


def read_dicom_directory_tree(path: Union[Path, Text]) -> Mapping:
    """Creates a dictionary with the available series and associated filenames."""

    path = str(Path(path) / "*.dcm")
    filenames = sorted(glob.glob(path))

    data_files: OrderedDict = OrderedDict()
    var_idx: Dict = {}
    for f in filenames:
        ds = pydicom.dcmread(f)

        if not parallel_spirals(ds):
            continue

        if ds.SeriesDescription not in data_files.keys():
            data_files[ds.SeriesDescription] = OrderedDict()
            var_idx = {}
            for var in VAR_OFFSET:
                data_files[ds.SeriesDescription][var] = []
                var_idx[int(Path(f).name[3:5]) + VAR_OFFSET[var]] = var

        var = var_idx[int(Path(f).name[3:5])]
        data_files[ds.SeriesDescription][var].append(f)

    return data_files


def parallel_spirals(dicom_data):
    """Checks if ParallelSpirals is in the tSequenceFileName."""
    csa = csar.get_csa_header(dicom_data, "series")
    try:
        ascii_header = csa["tags"]["MrPhoenixProtocol"]["items"][0]
        tSequenceFileName = re.search('tSequenceFileName\t = \t""(.*)""', ascii_header)
        return True if "ParallelSpirals" in tSequenceFileName[1] else False
    except TypeError:
        return False


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
    """Returns the images for a given series and variable."""
    if len(origin) == 0:
        return []

    assert series in origin
    assert variable in origin[series]

    images = []
    for f in origin[series][variable]:
        ds = pydicom.dcmread(f)
        images.append(ds.pixel_array)

    return images


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

    magnitude: ndarray
    phase: ndarray

    vector_axis: ClassVar[int] = 0
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

    def to_numpy(pattern: Text, data: Mapping) -> ndarray:
        from numpy import stack, argsort

        result = stack(
            tuple(
                stack(data[key]) for key in map(lambda x: pattern + x, ("X", "Y", "Z"))
            )
        )
        return result.transpose(
            argsort(
                (
                    ImageTimeSeries.vector_axis,
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
    """ Reads a StrainMap file with existing information on previous segmentations. """
    if str(filename).endswith(".h5"):
        return read_h5_file(filename)
    elif str(filename).endswith(".m"):
        return read_matlab_file(filename)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_h5_file(filename: Union[Path, Text]):
    """Reads a HDF5 file."""
    raise NotImplementedError


def read_matlab_file(filename: Union[Path, Text]):
    """Reads a Matlab file."""
    raise NotImplementedError

import glob
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Optional, Text, Union, Dict

import pydicom

VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


def read_dicom_directory_tree(path: Union[Path, Text]) -> Mapping:
    """ Creates a dictionary with the available series and associated filenames. """

    path = str(Path(path) / "*.dcm")
    filenames = sorted(glob.glob(path))

    data_files: OrderedDict = OrderedDict()
    var_idx: Dict = {}
    for f in filenames:
        ds = pydicom.dcmread(f)
        assert "SeriesDescription" in ds.dir()

        if ds.SeriesDescription not in data_files.keys():
            data_files[ds.SeriesDescription] = OrderedDict()
            var_idx = {}
            for var in VAR_OFFSET:
                data_files[ds.SeriesDescription][var] = []
                var_idx[int(Path(f).name[3:5]) + VAR_OFFSET[var]] = var

        var = var_idx[int(Path(f).name[3:5])]
        data_files[ds.SeriesDescription][var].append(f)

    return data_files


def read_dicom_file_tags(
    origin: Union[Mapping, Path, Text],
    series: Optional[Text] = None,
    variable: Optional[Text] = None,
    timestep: Optional[int] = None,
) -> Mapping:
    """ Returns a dictionary with the tags and values available in a DICOM file. """
    if isinstance(origin, Mapping):
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
    """ Returns the images for a given series and variable. """
    assert series in origin
    assert variable in origin[series]

    images = []
    for f in origin[series][variable]:
        ds = pydicom.dcmread(f)
        images.append(ds.pixel_array)

    return images


def read_all_images(origin: Mapping) -> Mapping:
    """ Returns all the images organised by series and variables. """
    images = deepcopy(origin)

    for series in images:
        for variable in images[series]:
            images[series][variable] = read_images(origin, series, variable)

    return images


def read_strainmap_file(filename: Union[Path, Text]):
    """ Reads a StrainMap file with existing information on previous segmentations. """
    if str(filename).endswith(".h5"):
        return read_h5_file(filename)
    elif str(filename).endswith(".m"):
        return read_matlab_file(filename)
    else:
        raise RuntimeError("File type not recognised by StrainMap.")


def read_h5_file(filename: Union[Path, Text]):
    """ Reads a HDF5 file. """
    raise NotImplementedError()


def read_matlab_file(filename: Union[Path, Text]):
    """ Reads a Matlab file. """
    raise NotImplementedError()

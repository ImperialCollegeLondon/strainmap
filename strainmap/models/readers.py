from __future__ import annotations

import glob
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path, PurePath, PurePosixPath
from typing import (
    Dict,
    List,
    Optional,
    Text,
    Tuple,
    Union,
    Sequence,
    Any,
)

import h5py
import numpy as np
import pandas as pd
import xarray as xr
import sparse
import pydicom
from natsort import natsorted
from functools import lru_cache

from .sm_data import LabelledArray


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


def phase_encoding(filename) -> tuple:
    """Indicates if X and Y Phases should be swapped and the velocity sign factors."""
    ds = pydicom.dcmread(filename)
    swap = ds.InPlanePhaseEncodingDirection == "ROW"
    signs = np.array([-1, 1, -1]) if swap else np.array([1, -1, 1])
    return swap, signs


def read_images(filenames: Sequence[str]) -> np.ndarray:
    """Returns the images for a given series and var."""
    return np.array([pydicom.dcmread(f).pixel_array for f in filenames])


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


def extract_strain_markers(
    h5file: Union[Path, h5py.File],
    datasets: Dict[str, str],
    regions: Dict[str, Sequence[str]],
    colnames: Sequence[str] = ("Slice", "Region", "Component", "PSS", "ESS", "PS",),
) -> pd.DataFrame:
    """Extracts as a DataFrame from a h5 file the strain markers.

    Only the datasets and regions requested are extracted and their names replaced by
    the corresponding values in the dictionaries. For regions, the entry for a given
    regional group should contain as many labels as regions in that group.

    Args:
        h5file (h5py.File): Opened h5File to extract the data from.
        datasets (Dict[str, str]): Dictionary with the datasets of interests and the
            replacing names.
        regions (Dict[str, Sequence[str]]): Dictionary with the regions of interest and
            the names of each of the individual regions.
        colnames (Sequence[str]): Sequence of names for the columns.

    Returns:
        Dataframe with the extracted data.
    """
    data: List[List[Any]] = []
    markers = (
        h5py.File(h5file, "r")["strain_markers"]
        if isinstance(h5file, Path)
        else h5file["strain_markers"]
    )

    # Loop over datasets
    for n, struct in markers.items():
        if n not in datasets:
            continue

        # Loop over regional groups (global, angular, etc.)
        for r, regdata in struct.items():
            if r not in regions:
                continue

            # Loop over the individual regions (global, AL, AS, I, etc.)
            for k, label in enumerate(regions[r]):

                # Loop over the components
                for j, comp in enumerate(("Longitudinal", "Radial", "Circumferential")):
                    data.append([datasets[n], label, comp])

                    # Loop over the PSS, ESS and PS markers
                    for i in range(regdata.shape[2]):
                        data[-1].append(regdata[k, j, i, 1])

    return pd.DataFrame(data, columns=colnames)


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
        """ Number of frames per dataset. """
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

    def phase_encoding(self, dataset: str) -> tuple:
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
        "mag",
        "x",
        "y",
        "z",
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

    vars = {"mag": "MagAvg", "z": "PhaseZ", "x": "PhaseX", "y": "PhaseY"}

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
    def sensitivity(self) -> np.ndarray:
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
                "comp": ["mag", "x", "y", "z"],
                "frame": np.arange(0, mag.shape[0]),
                "cine": dataset,
            },
        )


def labels_from_group(group: h5py.Group) -> Tuple[Sequence[str], Dict[str, Sequence]]:
    """ Reads the labels from a h5 group and return dimensions and coordinates. """
    dims: Sequence[str] = tuple(group.keys())
    coords: Dict[str, Sequence] = {
        d: None if group[d].shape is None else list(group[d][...].astype(np.str))
        for d in dims
    }
    return dims, coords


def group_to_labelled_array(group: h5py.Group) -> LabelledArray:
    """ Reads labels and values from a h5 group and constructs a labelled array.
    """
    assert "labels" in group, "Missing 'labels' information."
    assert "values" in group, "Missing 'values' information."

    if "fill_value" in group.attrs:
        data = group["values"][...]
        coords = group["coords"][...]
        shape = tuple(group.attrs["shape"][...])
        fill_value = group.attrs["fill_value"]
        values = sparse.COO(
            coords=coords, data=data, shape=shape, fill_value=fill_value
        )
    else:
        values = group["values"][...]

    dims, coords = labels_from_group(group["labels"])

    return LabelledArray(dims, coords, values)

import glob
from pathlib import Path

import pydicom

VAR_OFFSET = {"MagZ": 0, "PhaseZ": 1, "MagX": 2, "PhaseX": 3, "MagY": 4, "PhaseY": 5}


class StrainMapData(object):
    def __init__(self):

        self.data_files = {}
        self.bg_files = {}

    def clear(self):
        """ Clear all data from the data structure."""
        self.data_files = {}
        self.bg_files = {}

    @staticmethod
    def fill_filenames(path):
        """ Creates a dictionary with the available series and associated filenames. """

        path = str(Path(path) / "*.dcm")
        filenames = sorted(glob.glob(path))

        data_files = {}
        var_idx = {}
        for f in filenames:
            ds = pydicom.dcmread(f)
            assert "SeriesDescription" in ds.dir()

            if ds.SeriesDescription not in data_files.keys():
                data_files[ds.SeriesDescription] = {}
                var_idx = {}
                for var in VAR_OFFSET:
                    data_files[ds.SeriesDescription][var] = []
                    var_idx[int(Path(f).name[3:5]) + VAR_OFFSET[var]] = var

            var = var_idx[int(Path(f).name[3:5])]
            data_files[ds.SeriesDescription][var].append(f)

        return data_files

    def fill_data_files(self, path):
        """ Populates the dictionary of data files. """
        self.data_files = self.fill_filenames(path)
        return self.data_files

    def fill_bg_files(self, path):
        """ Populates the dictionary of background data. """
        self.bg_files = self.fill_filenames(path)
        return self.bg_files

    def get_images(self, series, variable):
        """ Returns the images data for a given series and variable. """
        data = []
        for f in self.data_files[series][variable]:
            ds = pydicom.dcmread(f)

            data.append(ds.pixel_array)

        return data

    def get_DICOM_file(self, series, variable, filenum):
        """ Returns a loaded DICOM file. """
        filename = self.data_files[series][variable][filenum]
        return pydicom.dcmread(filename)

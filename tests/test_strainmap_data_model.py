from pytest import mark
import sys


def test_from_folder(dicom_data_path):
    from strainmap.models.strainmap_data_model import StrainMapData

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files.files) > 0
    assert data.bg_files is None


def test_add_h5_file(dicom_data_path, tmpdir):
    from strainmap.models.strainmap_data_model import StrainMapData
    import h5py

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    data.add_h5_file(strainmap_file=tmpdir / "Dummy_file.h5")
    assert isinstance(data.strainmap_file, h5py.File)


def test_save(tmpdir, segmented_data):
    segmented_data.add_h5_file(strainmap_file=tmpdir / "Dummy_file.h5")

    segmented_data.segments["my_data"] = [1, 2, 3]
    segmented_data.save(["segments", "my_data"])
    s = "/".join(["segments", "my_data"])
    assert s in segmented_data.strainmap_file
    assert all(segmented_data.strainmap_file[s].value == [1, 2, 3])

    segmented_data.segments["my_data"] = [5, 2, 3]
    segmented_data.save(["segments", "my_data"])
    assert s in segmented_data.strainmap_file
    assert not all(segmented_data.strainmap_file[s].value == [1, 2, 3])


@mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
def test_from_file(dicom_data_path, h5_file_path):
    from strainmap.models.strainmap_data_model import StrainMapData
    from strainmap.models.readers import from_relative_paths

    data = StrainMapData.from_file(h5_file_path)
    assert isinstance(data, StrainMapData)
    assert data.data_files == ()

    data.add_paths(data_files=dicom_data_path)
    assert len(data.data_files.files) > 0

    for d, dataset in data.data_files.files.items():
        for v, variable in dataset.items():
            paths = from_relative_paths(
                h5_file_path, data.strainmap_file["data_files"][d][v][:]
            )
            assert variable == paths

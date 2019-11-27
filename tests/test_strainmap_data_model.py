def test_from_folder(dicom_data_path):
    from strainmap.models.strainmap_data_model import StrainMapData

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files) == 3
    assert len(data.bg_files) == 0


def test_add_paths(dicom_data_path, dicom_bg_data_path):
    from strainmap.models.strainmap_data_model import StrainMapData

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    data.add_paths(bg_files=dicom_bg_data_path)
    assert len(data.data_files) == 3
    assert len(data.bg_files) == 3


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

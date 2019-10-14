from pytest import mark
import sys


def test_factory_with_new_data(dicom_data_path, dicom_bg_data_path, tmpdir):
    from strainmap.models.strainmap_data_model import factory, StrainMapData
    import h5py

    data = factory(data_files=dicom_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files) == 3
    assert len(data.bg_files) == 0

    data = factory(data=data, bg_files=dicom_bg_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files) == 3
    assert len(data.bg_files) == 3

    data = factory(strainmap_file=tmpdir / "Dummy_file.h5")
    assert isinstance(data, StrainMapData)
    assert len(data.data_files) == 0
    assert len(data.bg_files) == 0
    assert isinstance(data.strainmap_file, h5py.File)

    data = factory(bg_files=dicom_bg_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files) == 0
    assert len(data.bg_files) == 0


@mark.skipif(
    sys.platform.startswith("win"),
    reason="Relative paths across units fail under Windows.",
)
def test_save(tmpdir, segmented_data):
    from strainmap.models.strainmap_data_model import factory

    data = factory(data=segmented_data, strainmap_file=tmpdir / "Dummy_file.h5")

    data.segments["my_data"] = [1, 2, 3]
    data.save(["segments", "my_data"])
    s = "/".join(["segments", "my_data"])
    assert s in data.strainmap_file
    assert all(data.strainmap_file[s].value == [1, 2, 3])

    data.segments["my_data"] = [5, 2, 3]
    data.save(["segments", "my_data"])
    assert s in data.strainmap_file
    assert not all(data.strainmap_file[s].value == [1, 2, 3])

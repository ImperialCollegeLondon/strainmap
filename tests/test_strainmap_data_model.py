from pytest import mark, approx


def test_from_folder(dicom_data_path):
    from strainmap.models.strainmap_data_model import StrainMapData

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    assert isinstance(data, StrainMapData)
    assert len(data.data_files.files) > 0


def test_add_h5_file(dicom_data_path, tmpdir):
    from strainmap.models.strainmap_data_model import StrainMapData
    import h5py

    data = StrainMapData.from_folder(data_files=dicom_data_path)
    data.add_h5_file(strainmap_file=tmpdir / "Dummy_file.h5")
    assert isinstance(data.strainmap_file, h5py.File)


def test_save(tmpdir, segmented_data):
    segmented_data.add_h5_file(strainmap_file=tmpdir / "Dummy_file.h5")

    cine = segmented_data.segments.cine[0].item()
    side = segmented_data.segments.side[0].item()
    segmented_data.segments.loc[{"cine": cine, "side": side}] = 42
    segmented_data.save(["segments"])
    s = f"/segments/{cine}/{side}"
    assert s in segmented_data.strainmap_file
    assert segmented_data.strainmap_file[s][()] == approx(42)

    segmented_data.segments.loc[{"cine": cine, "side": side}] = 24
    segmented_data.save(["segments"])
    assert s in segmented_data.strainmap_file
    assert not segmented_data.strainmap_file[s][()] == approx(42)


# @mark.skipif(sys.platform == "win32", reason="does not run on windows in Azure")
@mark.xfail
def test_from_file(dicom_data_path, h5_file_path):
    from strainmap.models.strainmap_data_model import StrainMapData
    from strainmap.models.readers import from_relative_paths

    data = StrainMapData.from_file(h5_file_path)
    assert isinstance(data, StrainMapData)
    assert data.data_files == ()

    data.add_paths(data_files=dicom_data_path)
    assert len(data.data_files.files) > 0

    for cine in data.data_files.files.cine.values:
        for comp in data.data_files.files.raw_comp.values:
            variable = data.data_files.files.sel(cine=cine, raw_comp=comp).values
            paths = from_relative_paths(
                h5_file_path, data.strainmap_file[f"/data_files/{cine}/{comp}"][:]
            )
            assert (variable == paths).all()

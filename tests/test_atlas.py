from unittest.mock import MagicMock, patch


def test_validate(dummy_data, tmp_path):
    from strainmap.gui.atlas_view import validate_data

    validated = validate_data(dummy_data)
    validated.equals(dummy_data)


def test_tabs(atlas_view):
    expected = ("Data",)

    assert len(atlas_view.notebook.tabs()) == len(expected)
    for i, t in enumerate(atlas_view.notebook.tabs()):
        assert atlas_view.notebook.tab(t, "text") == expected[i]


def test_load_atlas_data(atlas_view, dummy_data, tmp_path):
    from strainmap.gui.atlas_view import empty_data
    from pandas.testing import assert_frame_equal
    from pathlib import Path

    atlas_view.controller.progress = MagicMock()

    assert atlas_view.load_atlas_data().equals(empty_data())
    assert atlas_view.load_atlas_data(Path("my_data")).equals(empty_data())
    atlas_view.controller.progress.assert_called_once_with("Unknown file my_data.")

    filename = tmp_path / "my_data.csv"
    dummy_data.to_csv(filename, index=False)
    assert_frame_equal(
        atlas_view.load_atlas_data(filename), dummy_data, check_dtype=False
    )
    assert atlas_view.path == filename

    filename = tmp_path / "my_data.csv"
    dummy_data.to_csv(filename)
    assert atlas_view.load_atlas_data(filename).equals(empty_data())
    assert atlas_view.controller.progress.call_count == 4


def test_get_new_data(atlas_view, dummy_data):
    from collections import namedtuple

    new = dummy_data.loc[dummy_data["Record"] == dummy_data["Record"].min()].drop(
        ["Record", "Included", "pssGLS", "essGLS", "psGLS"], 1
    )
    atlas_view.controller.progress = MagicMock()

    with patch("strainmap.models.readers.extract_strain_markers", lambda *x, **y: new):
        atlas_view.controller.data = namedtuple(
            "Data", ["strain_markers", "strainmap_file", "orientation", "gls"]
        )(strain_markers=1, strainmap_file="", orientation="CW", gls=[1, 2, 3])
        actual = atlas_view.get_new_data()
        assert "Record" in actual.columns
        assert "Included" in actual.columns

    atlas_view.controller.data = None
    assert atlas_view.get_new_data() is None
    atlas_view.controller.progress.assert_called_once_with(
        "No patient data available. Load data to proceed."
    )


def test_add_record(atlas_view, dummy_data):
    new = dummy_data.loc[dummy_data["Record"] == dummy_data["Record"].min()]
    atlas_view.get_new_data = MagicMock(side_effect=[None, new])
    atlas_view.controller.progress = MagicMock()
    atlas_view.save_atlas = MagicMock()
    atlas_view.update_table = MagicMock()
    atlas_view.update_plots = MagicMock()
    atlas_view.overlay = MagicMock()

    atlas_view.add_record()
    atlas_view.controller.progress.assert_called_once_with(
        "No atlas file. Open an existing atlas or "
        "'Save atlas as' to create an empty one."
    )

    atlas_view.path = "my_way"
    before = len(atlas_view.atlas_data)
    atlas_view.add_record()
    assert len(atlas_view.atlas_data) == before

    atlas_view.add_record()
    assert len(atlas_view.atlas_data) == before + len(new)
    atlas_view.save_atlas.assert_called_once_with("my_way")
    atlas_view.overlay.assert_called_once_with(new)


def test_save_load_atlas(atlas_view, dummy_data, tmp_path):
    from pandas.testing import assert_frame_equal
    import pandas as pd

    atlas_view.controller.progress = MagicMock()
    atlas_view.atlas_data = dummy_data
    path = tmp_path / "my_way.csv"
    atlas_view.save_atlas(path)
    assert atlas_view.path == path

    atlas_view.atlas_data = pd.DataFrame()
    with patch("tkinter.filedialog.askopenfilename", lambda *x, **y: path):
        atlas_view.load_atlas()
    assert_frame_equal(atlas_view.atlas_data, dummy_data)


def test_remove_record(atlas_view_with_data):
    import numpy as np

    av = atlas_view_with_data

    num = np.random.randint(0, len(av.atlas_data))
    record = av.atlas_data.loc[num, "Record"]
    child_id = av.table.get_children()[num]
    av.table.selection_set(child_id)

    with patch("tkinter.messagebox.askyesno", lambda *x, **y: True):
        av.remove_record()

    assert record not in av.atlas_data.Record.values


def test_include_exclude_record(atlas_view_with_data):
    import numpy as np

    av = atlas_view_with_data

    num = np.random.randint(0, len(av.atlas_data))
    record = av.atlas_data.loc[num, "Record"]

    child_id = av.table.get_children()[num]
    av.table.selection_set(child_id)
    av.include_exclude_record()
    assert (~av.atlas_data.Included[av.atlas_data.Record == record]).all()

    child_id = av.table.get_children()[num]
    av.table.selection_set(child_id)
    av.include_exclude_record()
    assert (av.atlas_data.Included[av.atlas_data.Record == record]).all()


def test_include_exclude_selected(atlas_view_with_data):
    import numpy as np

    av = atlas_view_with_data

    num = np.random.randint(0, len(av.atlas_data))

    child_id = av.table.get_children()[num]
    av.table.selection_set(child_id)
    av.include_exclude_selected()
    assert not av.atlas_data.Included[num]

    child_id = av.table.get_children()[num]
    av.table.selection_set(child_id)
    av.include_exclude_record()
    assert av.atlas_data.Included[num]

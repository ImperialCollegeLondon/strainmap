from unittest.mock import MagicMock


def test_update_file_box(main_window, strainmap_data):
    from strainmap.gui.view_dicom_data import DICOMData

    dicom = DICOMData(main_window, strainmap_data)
    dicom.update_tree = MagicMock()

    dicom.nametowidget("series.seriesBox").event_generate("<<ComboboxSelected>>")
    assert dicom.update_tree.call_count == 1

    dicom.nametowidget("variables.variablesBox").event_generate("<<ComboboxSelected>>")
    assert dicom.update_tree.call_count == 2


def test_update_tree(main_window, strainmap_data):
    from strainmap.gui.view_dicom_data import DICOMData

    dicom = DICOMData(main_window, strainmap_data)
    dicom.treeview.insert = MagicMock()

    assert dicom.treeview.insert.call_count == 0
    dicom.nametowidget("files.filesBox").event_generate("<<ComboboxSelected>>")
    assert dicom.treeview.insert.call_count > 0

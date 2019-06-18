from unittest.mock import MagicMock


def test_creation(control_with_mock_window):
    from strainmap.main import Requisites

    assert control_with_mock_window.achieved == Requisites.NONE
    assert control_with_mock_window.window.add.call_count == 1

    control_with_mock_window.window.reset_mock()


def test_unlock_lock(control_with_mock_window):
    from strainmap.main import Requisites

    control_with_mock_window.unlock(Requisites.DATALOADED)
    assert control_with_mock_window.achieved == Requisites.NONE | Requisites.DATALOADED
    assert control_with_mock_window.window.add.call_count == 3

    control_with_mock_window.lock(Requisites.DATALOADED)
    assert control_with_mock_window.achieved == Requisites.NONE


def test_select_actions(control_with_mock_window):

    view_0 = control_with_mock_window.registered_views[0]
    view_1 = control_with_mock_window.registered_views[1]
    assert len(control_with_mock_window.select_actions(view_0)) == 2
    assert len(control_with_mock_window.select_actions(view_1)) == 0


def test_update_views(control_with_mock_window):

    view = control_with_mock_window.registered_views[0]
    control_with_mock_window.window.views = [MagicMock(view)]
    control_with_mock_window.update_views("Dummy")
    assert control_with_mock_window.data == "Dummy"
    assert control_with_mock_window.window.views[0].data == "Dummy"

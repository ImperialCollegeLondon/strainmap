def test_main_window(main_window, registered_views):

    assert len(main_window.views) == 0

    main_window.add(registered_views[0], None)
    main_window.add(registered_views[1], None)
    assert len(main_window.views) == 2
    assert all([v in registered_views for v in main_window.view_classes])

    main_window.remove(main_window.views[0])
    assert len(main_window.views) == 1


def test_requisites():
    from strainmap.gui.base_window_and_task import Requisites

    a = Requisites.NONE | Requisites.DATALOADED

    b = Requisites.DATALOADED
    assert Requisites.check(a, b)

    b = Requisites.SEGMENTED
    assert not Requisites.check(a, b)

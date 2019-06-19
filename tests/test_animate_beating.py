def test_launch_button(main_window, strainmap_data):
    from strainmap.gui.animate_beating import Animation

    beating = Animation(main_window, strainmap_data)

    assert not beating.anim
    beating.nametowidget("launchAnimation").invoke()
    assert beating.anim

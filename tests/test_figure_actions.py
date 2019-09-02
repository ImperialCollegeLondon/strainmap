from pytest import approx


def test_get_deltas(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import get_deltas

    last_event = MouseEvent("click", figure.canvas, x=100, y=200)
    event = MouseEvent("click", figure.canvas, x=200, y=200)

    deltax, deltay = get_deltas(event, last_event)

    assert deltax > 0
    assert deltay == 0


def test_zoom(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import ZoomAndPan

    zoom_and_pan = ZoomAndPan()
    last_event = MouseEvent("click", figure.canvas, x=100, y=200)
    event = MouseEvent("click", figure.canvas, x=200, y=200)

    xlim = last_event.inaxes.get_xlim()
    ylim = last_event.inaxes.get_ylim()

    zoom_and_pan.zoom(event, last_event)

    new_xlim = last_event.inaxes.get_xlim()
    new_ylim = last_event.inaxes.get_ylim()

    assert new_xlim[0] - xlim[0] < 0
    assert new_xlim[0] - xlim[0] == approx(xlim[1] - new_xlim[1])
    assert new_ylim[0] - ylim[0] < 0
    assert new_ylim[0] - ylim[0] == approx(ylim[1] - new_ylim[1])


def test_pan(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import ZoomAndPan

    zoom_and_pan = ZoomAndPan()
    last_event = MouseEvent("click", figure.canvas, x=100, y=200)
    event = MouseEvent("click", figure.canvas, x=200, y=100)

    xlim = last_event.inaxes.get_xlim()
    ylim = last_event.inaxes.get_ylim()

    zoom_and_pan.pan(event, last_event)

    new_xlim = last_event.inaxes.get_xlim()
    new_ylim = last_event.inaxes.get_ylim()

    assert new_xlim[0] - xlim[0] < 0
    assert new_xlim[0] - xlim[0] == approx(new_xlim[1] - xlim[1])
    assert new_ylim[0] - ylim[0] > 0
    assert new_ylim[0] - ylim[0] == approx(new_ylim[1] - ylim[1])


def test_reset(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import ZoomAndPan

    figure.axes[0].plot([1, 2, 3])

    zoom_and_pan = ZoomAndPan()
    event = MouseEvent("click", figure.canvas, x=200, y=100)

    xlim = event.inaxes.get_xlim()
    ylim = event.inaxes.get_ylim()

    event.inaxes.set_xlim(2 * xlim[0], 2 * xlim[1])
    event.inaxes.set_ylim(2 * ylim[0], 2 * ylim[1])

    zoom_and_pan.reset_zoom_and_pan(event)

    new_xlim = event.inaxes.get_xlim()
    new_ylim = event.inaxes.get_ylim()

    assert xlim == new_xlim
    assert ylim == new_ylim


def test_brightness_and_contrast(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import BrightnessAndContrast
    import numpy as np

    data = np.random.random((10, 10))

    b_and_c = BrightnessAndContrast()
    last_event = MouseEvent("click", figure.canvas, x=100, y=200)
    event = MouseEvent("click", figure.canvas, x=100, y=100)
    assert b_and_c.brightness_and_contrast(event, last_event) is None

    im = figure.axes[0].imshow(data)
    clim = im.get_clim()
    b_and_c.brightness_and_contrast(event, last_event)
    new_clim = im.get_clim()
    assert new_clim[1] - new_clim[0] == approx(clim[1] - clim[0])

    new_event = MouseEvent("click", figure.canvas, x=200, y=100)
    b_and_c.brightness_and_contrast(new_event, event)
    newer_clim = im.get_clim()
    assert newer_clim[1] - new_clim[1] == approx(new_clim[0] - newer_clim[0])


def test_reset_brightness_and_contrast(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import BrightnessAndContrast
    import numpy as np

    data = np.random.random((10, 10))
    im = figure.axes[0].imshow(data)
    clim = im.get_clim()
    im.set_clim(2 * clim[0], 2 * clim[1])

    b_and_c = BrightnessAndContrast()
    event = MouseEvent("click", figure.canvas, x=100, y=100)

    b_and_c.reset_brightness_and_contrast(event)
    reset_clim = im.get_clim()

    assert clim == reset_clim


def test_scroll_frames(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import ScrollFrames, data_scroller
    import numpy as np

    data = np.random.random((3, 10, 10))

    ax = figure.axes[0]
    ax.imshow(data[0])

    scroll = ScrollFrames()
    scroll.set_generators(data_scroller(data), axes=ax)
    event = MouseEvent("click", figure.canvas, x=100, y=100, step=1)

    scroll.show_frame_number(event)
    assert event.inaxes.images[0].get_array().data == approx(data[0])

    scroll.scroll_axes(event)
    assert event.inaxes.images[0].get_array().data == approx(data[1])


def test_animate(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import ScrollFrames
    import numpy as np

    ax = figure.axes[0]
    for i in range(3):
        ax.imshow(np.random.random((10, 10)))

    scroll = ScrollFrames()
    event = MouseEvent("click", figure.canvas, x=100, y=100)

    scroll.animate(event)
    assert ax in scroll._anim
    assert ax in scroll._anim_running
    assert scroll._anim[ax].event_source


def test_circle():
    from strainmap.gui.figure_actions import circle
    import numpy as np

    points = np.array([[0, 0], [1, 0]])

    expected = np.array([[0, -1, 0, 1, 0], [1, 0, -1, 0, 1]])
    actual = circle(points, resolution=4)

    assert expected == approx(actual)


def test_simple_closed_contour():
    from strainmap.gui.figure_actions import simple_closed_contour
    import numpy as np

    points = np.array([[0, 0], [1, 0], [1, -1]])

    expected = np.array([[0, 1, 1, 0], [0, 0, -1, 0]])
    actual = simple_closed_contour(points, points_per_contour=3)

    assert expected == approx(actual)


def test_spline():
    from strainmap.gui.figure_actions import spline
    import numpy as np

    points = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])

    actual = spline(points, points_per_contour=4, resolution=4)

    assert actual[0] == approx(actual[-1])
    assert actual[1] == approx(actual[-2] * np.array([1, -1]))


def test_add_point(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import DrawContours
    import numpy as np

    data = np.random.random((10, 10))
    axes = figure.axes[0]
    axes.imshow(data)

    draw = DrawContours()
    event = MouseEvent("click", figure.canvas, x=100, y=200)

    xdata = event.xdata
    ydata = event.ydata

    draw.add_point(None, event)
    assert len(draw.points[axes]) == 1
    assert len(draw.marks[axes]) == 1
    assert len(draw.contours[axes]) == 0
    assert draw.points[axes][0] == (xdata, ydata)


def test_add_contour(figure):
    from strainmap.gui.figure_actions import DrawContours
    import numpy as np

    data = np.random.random((10, 10))
    axes = figure.axes[0]
    axes.imshow(data)

    draw = DrawContours(num_contours=6)
    draw.points[axes].append((5, 5))
    draw.points[axes].append((5, 6))

    draw.add_contour(axes)

    assert len(draw.contours[axes]) == 1
    assert draw.num_points == 12


def test_remove_artist(figure):
    from matplotlib.backend_bases import MouseEvent, PickEvent
    from strainmap.gui.figure_actions import DrawContours
    import numpy as np

    data = np.random.random((10, 10))
    axes = figure.axes[0]
    axes.imshow(data)

    draw = DrawContours()
    event1 = MouseEvent("click", figure.canvas, x=100, y=200)
    event2 = MouseEvent("click", figure.canvas, x=100, y=250)

    draw.add_point(None, event1)
    draw.add_point(None, event2)
    assert len(draw.contours[axes]) == 1
    assert len(draw.marks[axes]) == 2

    pick = PickEvent("pick_event", figure.canvas, event2, draw.contours[axes][0])
    draw.remove_artist(None, pick)
    assert len(draw.contours[axes]) == 0

    pick = PickEvent("pick_event", figure.canvas, event1, draw.marks[axes][0])
    draw.remove_artist(None, pick)
    assert len(draw.marks[axes]) == 1


def test_clear_drawing(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import DrawContours
    import numpy as np

    data = np.random.random((10, 10))
    axes = figure.axes[0]
    axes.imshow(data)

    draw = DrawContours()
    event1 = MouseEvent("click", figure.canvas, x=100, y=200)
    event2 = MouseEvent("click", figure.canvas, x=100, y=250)

    draw.add_point(None, event1)
    draw.add_point(None, event2)
    assert len(draw.contours[axes]) == 1
    assert len(draw.marks[axes]) == 2

    draw.clear_drawing(event1)
    assert len(draw.points[axes]) == 0
    assert len(draw.marks[axes]) == 0
    assert len(draw.contours[axes]) == 0


def test_add_marker(figure):
    from strainmap.gui.figure_actions import Markers
    import numpy as np

    data = np.random.random((2, 10))
    axes = figure.axes[0]
    line = axes.plot(*data, label="my_data")[0]

    mark = Markers()
    mark.add_marker(line, label="my_marker")

    assert len(mark._linked_data) == 1
    assert list(mark._linked_data.keys())[0].get_label() == "my_marker"
    assert list(mark._linked_data.values())[0] == line


def test_update_marker_position(figure):
    from strainmap.gui.figure_actions import Markers
    import numpy as np

    data = np.random.random((2, 10))
    axes = figure.axes[0]
    line = axes.plot(*data, label="my_data")[0]

    mark = Markers()
    mark.add_marker(line, label="my_marker")
    mark.update_marker_position("my_marker", "my_data", 0.5)

    ind = np.argmin(np.abs(data[0] - 0.5))
    expected = data[1, ind]
    actual = list(mark._linked_data.keys())[0].get_ydata()[0]

    assert expected == approx(actual)


def test_get_closest(figure):
    from strainmap.gui.figure_actions import Markers
    import numpy as np

    data = np.random.random((2, 10))
    axes = figure.axes[0]
    line = axes.plot(*data, label="my_data")[0]

    mark = Markers()

    ind = np.argmin(np.abs(data[0] - 0.5))
    expected = data[0, ind], data[1, ind], ind
    actual = mark.get_closest(line, 0.5)

    assert expected == actual


def test_drag_marker(figure):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions import Markers
    import numpy as np

    def on_marker_moved(marker_name, data_name, x, y, idx):
        assert mark._current_marker.get_xdata()[0] == approx(x)
        assert mark._current_marker.get_ydata()[0] == approx(y)

    data = np.random.random((2, 10))
    axes = figure.axes[0]
    line = axes.plot(*data, label="my_data")[0]

    mark = Markers(on_marker_moved)
    mark.add_marker(line, label="my_marker")
    mark._current_marker = axes.lines[1]
    mark._current_data = line

    event = MouseEvent("click", figure.canvas, x=100, y=200)
    mark.drag_marker(event, event)

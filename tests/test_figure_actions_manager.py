from unittest.mock import MagicMock


def test_trigger_signature():
    from strainmap.gui.figure_actions_manager import (
        TriggerSignature,
        Location,
        MouseAction,
        Button,
    )

    s1 = TriggerSignature(Location.EDGE, Button.LEFT, MouseAction.MOVE)
    s2 = TriggerSignature(Location.N, Button.LEFT, MouseAction.MOVE)
    s3 = TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.MOVE)

    assert s2 in s1
    assert s3 not in s1


def test_fam_clean_events(actions_manager):
    actions_manager._event = ["an event", "another event"]
    actions_manager.clean_events()
    assert len(actions_manager._event) == 0


def test_mouse_clicked(actions_manager):
    from matplotlib.backend_bases import MouseEvent

    event = MouseEvent("clicked", actions_manager.canvas, 100, 200)
    actions_manager._on_mouse_pressed(event)
    assert len(actions_manager._event) == 1


def test_mouse_moved(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from time import sleep, time

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock(return_value=[])

    # Too soon to do anything
    actions_manager._time_init = time()
    actions_manager._on_mouse_moved(event)
    assert not actions_manager._select_action.called

    # No event to execute
    actions_manager._time_init = time()
    sleep(0.25)
    actions_manager._on_mouse_moved(event)
    assert actions_manager._select_action.called

    # One event to execute
    action1 = MagicMock()
    actions_manager._current_action = None

    actions_manager._select_action = MagicMock(return_value=[action1])
    actions_manager._time_init = time()
    sleep(0.25)
    actions_manager._on_mouse_moved(event)
    action1.assert_called_once_with(event, None)

    # More than one event to execute
    action1 = MagicMock()
    action2 = MagicMock()
    actions_manager._current_action = None

    actions_manager._select_action = MagicMock(return_value=[action1, action2])
    actions_manager._time_init = time()
    sleep(0.25)
    actions_manager._on_mouse_moved(event)
    action1.assert_called_once_with(event, None)
    action2.assert_called_once_with(event, None)


def test_mouse_released(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction
    from time import sleep, time

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock()
    actions_manager._select_click_type = MagicMock(
        return_value=(None, MouseAction.CLICK, event)
    )

    # Too late for a release event
    actions_manager._time_init = time()
    sleep(0.25)
    actions_manager._on_mouse_released(event)
    assert not actions_manager._select_action.called

    # A release event
    actions_manager._time_init = time()
    actions_manager._on_mouse_released(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.NONE, MouseAction.CLICK
    )


def test_mouse_scroll(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction

    event = MouseEvent("moved", actions_manager.canvas, 100, 200, button=2)
    actions_manager._select_action = MagicMock()

    actions_manager._on_mouse_scrolled(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.CENTRE, MouseAction.SCROLL
    )


def test_mouse_enter_axes(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock()

    actions_manager._on_entering_axes(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.NONE, MouseAction.ENTERAXES
    )


def test_mouse_leave_axes(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock()

    actions_manager._on_leaving_axes(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.NONE, MouseAction.LEAVEAXES
    )


def test_mouse_enter_figure(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock()

    actions_manager._on_entering_figure(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.NONE, MouseAction.ENTERFIGURE
    )


def test_mouse_leave_figure(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location, Button, MouseAction

    event = MouseEvent("moved", actions_manager.canvas, 100, 200)
    actions_manager._select_action = MagicMock()

    actions_manager._on_leaving_figure(event)
    actions_manager._select_action.assert_called_once_with(
        Location.OUTSIDE, Button.NONE, MouseAction.LEAVEFIGURE
    )


def test_select_click_type(actions_manager):
    from matplotlib.backend_bases import MouseEvent, PickEvent
    from strainmap.gui.figure_actions_manager import MouseAction

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    pickevent = PickEvent("pick_event", actions_manager.canvas, event, None)
    actions_manager._event = [event, pickevent]
    ev, mouse_action, mouse_event = actions_manager._select_click_type()
    assert mouse_action == MouseAction.PICK
    assert ev == pickevent
    assert mouse_event == event

    event = MouseEvent("click", actions_manager.canvas, 100, 200, dblclick=True)
    pickevent = PickEvent("pick_event", actions_manager.canvas, event, None)
    actions_manager._event = [event, pickevent]
    ev, mouse_action, mouse_event = actions_manager._select_click_type()
    assert mouse_action == MouseAction.DPICK
    assert ev == pickevent
    assert mouse_event == event

    event = MouseEvent("click", actions_manager.canvas, 100, 200, dblclick=True)
    actions_manager._event = [event]
    ev, mouse_action, mouse_event = actions_manager._select_click_type()
    assert mouse_action == MouseAction.DCLICK
    assert ev == event
    assert mouse_event == event

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    actions_manager._event = [event]
    ev, mouse_action, mouse_event = actions_manager._select_click_type()
    assert mouse_action == MouseAction.CLICK
    assert ev == event
    assert mouse_event == event


def test_select_movement_type(actions_manager):
    from matplotlib.backend_bases import MouseEvent, PickEvent
    from strainmap.gui.figure_actions_manager import MouseAction

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    actions_manager._event = []
    ev, mouse_action, mouse_event = actions_manager._select_movement_type(event)
    assert mouse_action == MouseAction.MOVE
    assert ev is None
    assert mouse_event == event

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    pickevent = PickEvent("pick_event", actions_manager.canvas, event, None)
    actions_manager._event = [event, pickevent]
    ev, mouse_action, mouse_event = actions_manager._select_movement_type(event)
    assert mouse_action == MouseAction.PICKDRAG
    assert ev == pickevent
    assert mouse_event == event

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    actions_manager._event = [event]
    ev, mouse_action, mouse_event = actions_manager._select_movement_type(event)
    assert mouse_action == MouseAction.DRAG
    assert ev == event
    assert mouse_event == event


def test_select_location(actions_manager):
    from matplotlib.backend_bases import MouseEvent
    from strainmap.gui.figure_actions_manager import Location

    actions_manager._figure().add_subplot()

    event = MouseEvent("click", actions_manager.canvas, 0, 0)
    location = actions_manager._select_location(event)
    assert location == Location.OUTSIDE

    event = MouseEvent("click", actions_manager.canvas, 100, 200)
    location = actions_manager._select_location(event)
    assert location != Location.OUTSIDE


def test_select_action(actions_manager):
    from strainmap.gui.figure_actions_manager import (
        TriggerSignature,
        Location,
        MouseAction,
        Button,
    )

    s1 = TriggerSignature(Location.EDGE, Button.LEFT, MouseAction.MOVE)
    s2 = TriggerSignature(Location.N, Button.LEFT, MouseAction.MOVE)
    s3 = TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.MOVE)

    actions_manager._actions[s1].append(None)
    actions_manager._actions[s2].append(None)
    actions_manager._actions[s3].append(None)

    actions = actions_manager._select_action(
        Location.OUTSIDE, Button.LEFT, MouseAction.MOVE
    )
    assert len(actions) == 0

    actions = actions_manager._select_action(
        Location.CENTRE, Button.LEFT, MouseAction.MOVE
    )
    assert len(actions) == 1

    actions = actions_manager._select_action(Location.S, Button.LEFT, MouseAction.MOVE)
    assert len(actions) == 1

    actions = actions_manager._select_action(Location.N, Button.LEFT, MouseAction.MOVE)
    assert len(actions) == 2


def test_add_remove_action(actions_manager, action):

    actions_manager.add_action(action)

    assert action.__name__ in actions_manager.__dict__
    for s in action().signatures.keys():
        assert s in actions_manager._actions

    actions_manager.remove_action(action.__name__)

    assert action.__name__ not in actions_manager.__dict__
    for s in action().signatures.keys():
        assert s not in actions_manager._actions


def test_get_mouse_location():
    from strainmap.gui.figure_actions_manager import Location, get_mouse_location

    limits = (0, 1, 0, 1)
    fraction = 0.2

    assert get_mouse_location(0.1, 0.1, *limits, fraction) == Location.NW
    assert get_mouse_location(0.9, 0.1, *limits, fraction) == Location.NE
    assert get_mouse_location(0.1, 0.9, *limits, fraction) == Location.SW
    assert get_mouse_location(0.9, 0.9, *limits, fraction) == Location.SE
    assert get_mouse_location(0.5, 0.1, *limits, fraction) == Location.N
    assert get_mouse_location(0.5, 0.9, *limits, fraction) == Location.S
    assert get_mouse_location(0.1, 0.5, *limits, fraction) == Location.W
    assert get_mouse_location(0.9, 0.5, *limits, fraction) == Location.E
    assert get_mouse_location(0.5, 0.5, *limits, fraction) == Location.CENTRE

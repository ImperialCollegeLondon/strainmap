from __future__ import annotations

import weakref
from collections import defaultdict
from enum import Flag, auto
from itertools import chain
from time import time
from typing import Callable, Dict, List, NamedTuple, Optional, Text, Type

import numpy as np
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure


class Location(Flag):
    N = auto()
    S = auto()
    E = auto()
    W = auto()
    NW = auto()
    NE = auto()
    SW = auto()
    SE = auto()
    CENTRE = auto()
    EDGE = N | S | E | W | NW | NE | SW | SE
    CROSS = CENTRE | N | S | E | W
    ANY = CENTRE | EDGE
    OUTSIDE = ~ANY


class Button(Flag):
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    CENTRE = auto()


class MouseAction(Flag):
    MOVE = auto()
    DRAG = auto()
    CLICK = auto()
    DCLICK = auto()
    SCROLL = auto()
    PICK = auto()
    DPICK = auto()
    PICKDRAG = auto()
    ENTERAXES = auto()
    LEAVEAXES = auto()
    ENTERFIGURE = auto()
    LEAVEFIGURE = auto()
    RELEASE = auto()


class TriggerSignature(NamedTuple):
    """Contains the signature triad needed to trigger an event.

    - location (Location): The location where the pointer must be
    - button (Button): The button that must be pressed
    - action (MouseAction): The action that is carried
    """

    location: Location
    button: Button
    mouse_action: MouseAction

    @property
    def locations_contained(self):
        """Provides all the locations that are contained by this one.

        For example, if a signature has Location.EDGE, a click event that takes
        place at Location.N and another that takes place at Location.S should both
        trigger the same action, as N and S are contained by EDGE.
        """
        return [loc for loc in Location if self.location & loc == loc]

    def __contains__(self, other) -> bool:
        if not isinstance(other, TriggerSignature):
            raise TypeError("Only a TriggerSignature can be assert equal with another.")

        return (
            other.location in self.locations_contained
            and other.button == self.button
            and other.mouse_action == self.mouse_action
        )


class ActionBase(object):
    """Base class for the actions.

    It ensures that all actions will have a signatures attribute containing all triggers
    relevant for the action.
    """

    def __init__(
        self,
        signatures: Dict[TriggerSignature, Callable[[Event, Event], Optional[Event]]],
    ):
        self.signatures = signatures


class FigureActionsManager(object):
    """Adds some interactivity functionality to a Matplotlib figure.

    This class adds interactive functionality to a standard figure, replacing
    the actions toolbar, by using only different mouse gestures happening in different
    areas of the axes (indeed, using it in combination with the toolbar might have
    strange results).

    The figure updated with the interactive functionality can be used normally as with
    any other figure. If the figure is to be used embedded in a GUI framework (e.g.
    Tkinter, Kivy, QT...), adding the interactive functionality must be done AFTER the
    figure has been added to the GUI.

    By default, the FigureActionsManager does not add any extra functionality. This
    can be included later on with the self.add_action method or during creation by
    providing the Actions as extra positional arguments. Arguments to the Actions
    can be passed as kwargs dictionaries using: options_ActionName = {}.

    For example, to add ZoomAndPan functionality to figure 'fig' and the ability to
    draw contours with certain options, one could do:

    fam = FigureActionsManager( fig, ZoomAndPan, DrawContours,
                                options_DrawContours=dict(num_contours=2 )
                                )
    """

    MOUSE_BUTTONS = {
        1: Button.LEFT,
        2: Button.CENTRE,
        3: Button.RIGHT,
        "up": Button.CENTRE,
        "down": Button.CENTRE,
    }

    def __init__(self, figure: Figure, *args, axis_fraction=0.2, delay=0.2, **kwargs):
        """An existing Matplotlib figure is the only input needed by the Manager.

        Args:
            figure: An instance of a Matplotlib Figure.
            *args: Actions that need to be added.
            axis_fraction: Fraction of the axes that define the edge on each side.
            delay: Time delay used to differentiate clicks from drag events.
            **kwargs: Parameters to be passed during to the creation of the Actions.
        """
        self.axis_fraction = np.clip(axis_fraction, 0, 0.5)
        self.delay = max(delay, 0)

        self._figure = weakref.ref(figure)
        self._time_init = 0
        self._event: List = []
        self._last_event = None
        self._current_action: list = []
        self._actions: Dict = defaultdict(list)

        self._connect_events()

        for action in args:
            self.add_action(action, **kwargs)

    @property
    def canvas(self):
        """The canvas this interaction is connected to."""
        return self._figure().canvas

    def draw(self):
        """Convenience method for re-drawing the canvas."""
        self.canvas.draw_idle()

    def add_action(self, action: Type[ActionBase], **kwargs):
        """Adds an action to the Manager."""
        options = kwargs.get("options_" + action.__name__, {})
        acc = action(**options)
        for k, v in acc.signatures.items():
            self._actions[k].append(v)
        self.__dict__[action.__name__] = acc

    def remove_action(self, action_name: Text):
        """Removes an action from the Manager."""
        for k, v in self.__dict__[action_name].signatures.items():
            for act in self._actions[k]:
                if act == v:
                    self._actions[k].remove(act)
            if len(self._actions[k]) == 0:
                self._actions.pop(k)
        del self.__dict__[action_name]

    def clean_events(self):
        """Removes all information related to previous events."""
        self._event = []
        self._last_event = None
        self._current_action = []

    def _connect_events(self):
        """Connects the relevant events to the canvas."""
        self.canvas.mpl_connect("button_press_event", self._on_mouse_pressed)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_released)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_moved)
        self.canvas.mpl_connect("scroll_event", self._on_mouse_scrolled)
        self.canvas.mpl_connect("axes_enter_event", self._on_entering_axes)
        self.canvas.mpl_connect("axes_leave_event", self._on_leaving_axes)
        self.canvas.mpl_connect("figure_enter_event", self._on_entering_figure)
        self.canvas.mpl_connect("figure_leave_event", self._on_leaving_figure)
        self.canvas.mpl_connect("pick_event", self._on_mouse_pressed)

    def _on_mouse_pressed(self, event):
        """Initial response to the click events by triggering the timer.

        After pressing a mouse button, several things might happen:

        1- The button is released in a time defined by self.delay. In this case,
            it is recorded as a clicked event and some action happens, which might be
            a single click, a double click or a pick action.
        2- The button is released but it takes longer. The clicked event is lost and
            nothing happens.
        3- The mouse is dragged while clicked. After self.delay, the action
            associated with that dragging is executed.
        """
        self._event.append(event)
        self._time_init = time()

    def _on_mouse_moved(self, event):
        """Runs actions related to moving the mouse over the figure."""
        if time() - self._time_init < self.delay:
            return

        elif len(self._current_action) == 0:
            self._last_event, mouse_action, mouse_event = self._select_movement_type(
                event
            )
            button = self.MOUSE_BUTTONS.get(mouse_event.button, Button.NONE)
            location = self._select_location(mouse_event)

            self._current_action = self._select_action(location, button, mouse_action)

        if len(self._current_action) == 0:
            return
        elif len(self._current_action) == 1:
            self._last_event = self._current_action[0](event, self._last_event)
        else:
            for ac in self._current_action:
                ac(event, self._last_event)

            self._current_action = []

        self.draw()

    def _on_mouse_released(self, event):
        """Stops the timer and executes the relevant events, if necessary."""
        if time() - self._time_init > self.delay:
            self.clean_events()
            ev = event
            mouse_action = MouseAction.RELEASE
            button = self.MOUSE_BUTTONS.get(event.button, Button.NONE)
            location = self._select_location(event)
        else:
            ev, mouse_action, mouse_event = self._select_click_type()
            button = self.MOUSE_BUTTONS.get(mouse_event.button, Button.NONE)
            location = self._select_location(mouse_event)

        self._execute_action_and_redraw(event, ev, location, button, mouse_action)

    def _on_mouse_scrolled(self, event):
        """Executes scroll events."""
        mouse_action = MouseAction.SCROLL
        button = Button.CENTRE
        location = self._select_location(event)

        self._execute_action_and_redraw(event, None, location, button, mouse_action)

    def _on_entering_axes(self, event):
        """Executes the actions related to entering a new axes."""
        mouse_action = MouseAction.ENTERAXES
        button = Button.NONE
        location = self._select_location(event)

        self._execute_action_and_redraw(event, None, location, button, mouse_action)

    def _on_leaving_axes(self, event):
        """Executes the actions related to leaving an axes."""
        mouse_action = MouseAction.LEAVEAXES
        button = Button.NONE
        location = self._select_location(event)

        self._execute_action_and_redraw(event, None, location, button, mouse_action)

    def _on_entering_figure(self, event):
        """Executes the actions related to entering the figure."""
        mouse_action = MouseAction.ENTERFIGURE
        button = Button.NONE
        location = Location.OUTSIDE

        self._execute_action_and_redraw(event, None, location, button, mouse_action)

    def _on_leaving_figure(self, event):
        """Executes the actions related to leaving a figure."""
        mouse_action = MouseAction.LEAVEFIGURE
        button = Button.NONE
        location = Location.OUTSIDE

        self._execute_action_and_redraw(event, None, location, button, mouse_action)

    def _execute_action_and_redraw(
        self, event, last_event, location, button, mouse_action
    ):
        """Execute one or more actions and redraws the plot."""
        action = self._select_action(location, button, mouse_action)

        for ac in action:
            ac(event, last_event)

        self.clean_events()
        self.draw()

    def _select_click_type(self):
        """Select the type of click.

        Here we need to discriminate between single clicks, double clicks and pick
        events (which might also generate a single click).
        """
        if len([p for p in self._event if p.name == "pick_event"]) > 0:
            ev = [p for p in self._event if p.name == "pick_event"][-1]
            if ev.mouseevent.dblclick:
                mouse_action = MouseAction.DPICK
            else:
                mouse_action = MouseAction.PICK
            mouse_event = ev.mouseevent

        elif self._event[-1].dblclick:
            mouse_event = ev = self._event[-1]
            mouse_action = MouseAction.DCLICK

        else:
            mouse_event = ev = self._event[-1]
            mouse_action = MouseAction.CLICK

        return ev, mouse_action, mouse_event

    def _select_movement_type(self, event):
        """Select the type of movement.

        Here we need to discriminate between just move, drag and pickdrag.
        """
        if len(self._event) == 0:
            mouse_action = MouseAction.MOVE
            mouse_event = event
            ev = None
        elif len([p for p in self._event if p.name == "pick_event"]) > 0:
            ev = [p for p in self._event if p.name == "pick_event"][-1]
            mouse_action = MouseAction.PICKDRAG
            mouse_event = ev.mouseevent
        else:
            mouse_action = MouseAction.DRAG
            mouse_event = ev = self._event[-1]

        return ev, mouse_action, mouse_event

    def _select_location(self, event) -> Location:
        """Select the type of location."""
        if event.inaxes is None:
            location = Location.OUTSIDE
        else:
            x, y = event.xdata, event.ydata
            xmin, xmax = sorted(event.inaxes.get_xlim())
            ymin, ymax = sorted(event.inaxes.get_ylim())
            location = get_mouse_location(
                x, y, xmin, xmax, ymin, ymax, self.axis_fraction
            )

        return location

    def _select_action(self, location, button, mouse_action):
        """Select the action to execute based on the received trigger signature.

        There might be several matching signatures. In that case, a list of actions
        is returned. Whether all actions are executed or not depends on the type of
        action.

        All actions are executed if:
            - Relate to simple mouse movement
            - Relate to clicks or picks

        Only the first action is executed if:
            - Relate to drag events (normal DRAG and PICKDRAG)
        """
        trigger = TriggerSignature(location, button, mouse_action)

        return list(
            chain.from_iterable(
                [
                    action
                    for signature, action in self._actions.items()
                    if trigger in signature
                ]
            )
        )


def get_mouse_location(x, y, xmin, xmax, ymin, ymax, fraction):
    """Assigns a logical location based on where the mouse is."""
    deltax = abs(xmax - xmin) * fraction
    deltay = abs(ymax - ymin) * fraction

    if xmin <= x <= xmin + deltax:
        if ymin <= y <= ymin + deltay:
            location = Location.NW
        elif ymax - deltay <= y <= ymax:
            location = Location.SW
        else:
            location = Location.W
    elif xmax - deltax <= x <= xmax:
        if ymin <= y <= ymin + deltay:
            location = Location.NE
        elif ymax - deltay <= y <= ymax:
            location = Location.SE
        else:
            location = Location.E
    else:
        if ymin <= y <= ymin + deltay:
            location = Location.N
        elif ymax - deltay <= y <= ymax:
            location = Location.S
        else:
            location = Location.CENTRE

    return location


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fam = FigureActionsManager(fig)

    assert fam == fig.actions_manager

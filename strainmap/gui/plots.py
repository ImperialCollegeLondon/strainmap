import weakref
from collections import deque
from enum import Flag, auto
from functools import partial
from time import time
from typing import Callable, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Location(Flag):
    CENTRE = auto()
    EDGE = auto()
    NW = auto()
    NE = auto()
    SW = auto()
    SE = auto()
    ANY = auto()


class Button(Flag):
    LEFT = auto()
    RIGHT = auto()
    CENTRE = auto()
    SCROLL = auto()


class Action(Flag):
    DRAG = auto()
    CLICK = auto()
    SCROLL = auto()


ACTIONS: Dict = dict()
"""Registry of the available mouse actions."""

MOUSE_BUTTONS = {
    1: Button.LEFT,
    2: Button.CENTRE,
    3: Button.RIGHT,
    "up": Button.SCROLL,
    "down": Button.SCROLL,
}
"""Translates the event.button information into an enumeration."""


def register_action(
    fun: Optional[Callable] = None,
    location: Optional[Location] = None,
    button: Optional[Button] = None,
    action: Optional[Action] = None,
):
    """Register the active areas of the figure with different actions.

    Compound areas like ANY or EDGE need to be added with all their possibilities.
    """
    if fun is None:
        return partial(register_action, location=location, button=button, action=action)

    if location is Location.EDGE:
        for loc in (Location.EDGE, Location.NW, Location.SW, Location.NE, Location.SE):
            ACTIONS[(loc, button, action)] = fun
    elif location is Location.ANY:
        for loc in (
            Location.EDGE,
            Location.CENTRE,
            Location.NW,
            Location.SW,
            Location.NE,
            Location.SE,
        ):
            ACTIONS[(loc, button, action)] = fun
    else:
        ACTIONS[(location, button, action)] = fun

    return fun


class InteractivePlot(object):
    """Adds some interactivity functionality to a matplotlib figure.

    This class adds several interactive functionality to a standard figure, replacing
    (to some extent) the actions toolbar by different mouse gestures happening in
    different areas of the axes.

    The functionality it adds is:

    - pan: when left click&drag starting around the center of the plot.
    - zoom: when left click&drag starting around the edges of the plot.
    - change brightness: right click&drag in the vertical direction.
    - change contrast: right click&drag in the horizontal direction.
    - scroll images: change between images in a plot with several images overlayed.

    The active area of the first two is controlled by the 'axis_fraction' input
    parameter. Brightness and contrast controls is only active when the plot contains an
    image.

    Example:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from strainmap.gui.plots import InteractivePlot
        >>> data = np.random.random((100, 100))
        >>> fig = plt.figure()
        >>> InteractivePlot(fig)
        >>> ax = fig.add_subplot()
        >>> ax.imshow(data, cmap=plt.get_cmap("binary"))

    The figure updated with the interactive functionality can be used normally as with
    any other figure.

    If the figure is to be used embedded in a GUI framework (e.g. Tkinter, Kivy, QT...),
    adding the interactive functionality must be done AFTER the figure has been added
    to the GUI.
    """

    def __init__(self, figure: Figure, axis_fraction=0.2, delta_time=0.2):
        self.axis_fraction = axis_fraction
        self.delta_time = delta_time
        self._canvas = weakref.ref(figure.canvas)
        self._time_init = 0
        self._img_shift = 0
        self._event = None
        self._action = None

        # Connect the relevant events
        self.canvas.mpl_connect("button_press_event", self._on_mouse_clicked)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_released)
        self.move_event_id = self.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_moved
        )
        self.canvas.mpl_connect("scroll_event", self._on_mouse_scrolled)
        self.canvas.mpl_connect("axes_enter_event", self._on_entering_axes)

        figure.interactions = self

    @property
    def canvas(self):
        """The canvas this interaction is connected to."""
        return self._canvas()

    def _on_mouse_clicked(self, event):
        """Triggers the timer.

        After clicking a mouse button, several things might happen:

        1- The button is released in a time defined by self.delta_time. In this case,
            it is recorded as a clicked event and some action happens, which might be
            a single click or a double click action.
        2- The button is released but it takes longer. The clicked event is lost and
            nothing happens.
        3- The mouse is dragged while clicked. After self.delta_time, the action
            associated with that dragging is executed.
        """
        if event.inaxes is None:
            return

        self._event = event
        self.time_init = time()

    def _on_mouse_released(self, event):
        """Stops the timer and executes a clicked event, if necessary."""
        if event.inaxes is None:
            return
        elif time() - self.time_init > self.delta_time:
            self._event = None
            self._action = None
            return

        location = self._get_mouse_location(self._event.xdata, self._event.ydata)
        button = MOUSE_BUTTONS[self._event.button]
        action = Action.CLICK

        ACTIONS.get((location, button, action), lambda *args: None)(self, self._event)

        self._event = None
        self._action = None

    def _on_mouse_moved(self, event):
        """Runs actions related to moving the mouse over the figure.

        Since this can happen for a variety of reasons and under different circumstances
        it is necessary to consider several cases separately.

        - If the mouse is in the figure but outside an axes, nothing happens.
        - If there is not an ongoing event due to a previous click or if the time since
            the click is too short, again nothing happens.
        - Assuming we got this far, if there is no action configured for this movement,
            one is configured depending on where the original click event happen.
        - Otherwise, the configured action is executed.
        """
        if event.inaxes is None:
            return
        elif self._event is None or time() - self.time_init < self.delta_time:
            return
        elif self._action is None:
            location = self._get_mouse_location(self._event.xdata, self._event.ydata)
            button = MOUSE_BUTTONS[self._event.button]
            action = Action.DRAG
            self._action = ACTIONS.get((location, button, action), lambda *args: None)

        self._action(self, event)

    def _on_mouse_scrolled(self, event):
        """The images available in the axes, if more than one, are scrolled."""
        if event.inaxes is None:
            return

        img = deque(event.inaxes.images)
        if len(img) == 0:
            return

        self._img_shift += event.step
        self._img_shift = self._img_shift % len(img)
        event.inaxes.set_title(f"Cine frame: {self._img_shift}", loc="left")
        img.rotate(event.step)
        event.inaxes.images = list(img)

        self.canvas.draw()

    def _on_entering_axes(self, event):
        """Shows the cine frame number on top of the axes."""
        if len(event.inaxes.images) > 0:
            event.inaxes.set_title(f"Cine frame: {self._img_shift}", loc="left")
            self.canvas.draw()

    def _get_mouse_location(self, x, y):
        """Assigns a logical location based on where the mouse was."""
        xmin, xmax = sorted(self._event.inaxes.get_xlim())
        ymin, ymax = sorted(self._event.inaxes.get_ylim())

        deltax = abs(xmax - xmin) * self.axis_fraction
        deltay = abs(ymax - ymin) * self.axis_fraction

        if xmin <= x <= xmin + deltax:
            if ymin <= y <= ymin + deltay:
                location = Location.NW
            elif ymax - deltay <= y <= ymax:
                location = Location.SW
            else:
                location = Location.EDGE
        elif xmax - deltax <= x <= xmax:
            if ymin <= y <= ymin + deltay:
                location = Location.NE
            elif ymax - deltay <= y <= ymax:
                location = Location.SE
            else:
                location = Location.EDGE
        else:
            if ymin <= y <= ymin + deltay:
                location = Location.EDGE
            elif ymax - deltay <= y <= ymax:
                location = Location.EDGE
            else:
                location = Location.CENTRE

        return location

    def get_deltas(self, event):
        pixel_to_data = self._event.inaxes.transData.inverted()
        data = pixel_to_data.transform_point((event.x, event.y))
        last_data = pixel_to_data.transform_point((self._event.x, self._event.y))

        deltax = data[0] - last_data[0]
        deltay = data[1] - last_data[1]

        return deltax, deltay

    @register_action(location=Location.EDGE, button=Button.LEFT, action=Action.DRAG)
    def _zoom(self, event):
        """Implements the zoom in and out functionality."""
        deltax, deltay = self.get_deltas(event)

        # Find relative displacement in x
        xlim = self._event.inaxes.get_xlim()
        xspan = xlim[1] - xlim[0]
        deltax = deltax / xspan

        # Find relative displacement in y
        ylim = self._event.inaxes.get_ylim()
        yspan = ylim[1] - ylim[0]
        deltay = deltay / yspan

        delta = np.sign(deltax + deltay) * np.sqrt(deltax ** 2 + deltay ** 2)

        # Update axis limits
        new_xlim = (xlim[0] - delta * xspan, xlim[1] + delta * xspan)
        new_ylim = (ylim[0] - delta * yspan, ylim[1] + delta * yspan)

        self._event.inaxes.set_xlim(new_xlim)
        self._event.inaxes.set_ylim(new_ylim)
        self.canvas.draw()

        self._event = event

    @register_action(location=Location.CENTRE, button=Button.LEFT, action=Action.DRAG)
    def _pan(self, event):
        """Pan functionality."""
        deltax, deltay = self.get_deltas(event)

        # Pan in x direction
        xlim = self._event.inaxes.get_xlim()
        ylim = self._event.inaxes.get_ylim()

        new_xlim = xlim[0] - deltax, xlim[1] - deltax
        new_ylim = ylim[0] - deltay, ylim[1] - deltay

        self._event.inaxes.set_xlim(new_xlim)
        self._event.inaxes.set_ylim(new_ylim)
        self.canvas.draw()

        self._event = event

    @register_action(location=Location.ANY, button=Button.RIGHT, action=Action.DRAG)
    def _brigthness_and_contrast(self, event):
        """Controls the brightness and contrast of an image.

        If there are more than one image in the axes, the one on top (the last one on
        the list) is used.
        """
        if len(self._event.inaxes.get_images()) == 0:
            return

        deltax, deltay = self.get_deltas(event)

        # Find relative displacement in x
        xlim = self._event.inaxes.get_xlim()
        xspan = xlim[1] - xlim[0]
        deltax = deltax / xspan

        # Find relative displacement in y
        ylim = self._event.inaxes.get_ylim()
        yspan = ylim[1] - ylim[0]
        deltay = deltay / yspan

        clim = self._event.inaxes.get_images()[-1].get_clim()
        cspan = clim[1] - clim[0]

        # Movement in Y controls the brightness
        clim_low, clim_high = clim[0] + cspan * deltay, clim[1] + cspan * deltay

        # Movement in X controls the contrast
        clim_low, clim_high = clim_low + cspan * deltax, clim_high - cspan * deltax

        self._event.inaxes.get_images()[-1].set_clim(clim_low, clim_high)
        self.canvas.draw()

        self._event = event

    @register_action(location=Location.SW, button=Button.LEFT, action=Action.CLICK)
    def _reset_zoom_and_pan(self, event):
        """Resets the axis limits to the original ones, before any zoom and pan."""
        self._event.inaxes.relim()
        self._event.inaxes.autoscale()
        self.canvas.draw()

    @register_action(location=Location.NW, button=Button.LEFT, action=Action.CLICK)
    def _reset_brighness_and_contrast(self, event):
        """Resets the brightness and contrast to the limits based on the data."""
        array = self._event.inaxes.get_images()[-1].get_array()
        clim_low, clim_high = array.min(), array.max()
        self._event.inaxes.get_images()[-1].set_clim(clim_low, clim_high)
        self.canvas.draw()


if __name__ == "__main__":
    import numpy as np

    # from pprint import pprint

    data = np.random.random((100, 100))
    data2 = np.random.random((100, 100))

    # Using Pyplot
    fig = plt.figure()
    InteractivePlot(fig)
    ax = fig.add_subplot()
    ax.imshow(data, cmap=plt.get_cmap("binary"))
    ax.imshow(data2, cmap=plt.get_cmap("binary"))
    plt.show()

    # pprint(ax.properties())
    # print(ax.images)
    # ax.images[0], ax.images[1] = ax.images[1], ax.images[0]
    # print(ax.images)

    # In Tkinter
    # import tkinter
    # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    #
    # fig = InteractivePlot.new_figure(
    #     figure_canvas=FigureCanvasTkAgg, master=tkinter.Tk()
    # )
    # fig.canvas.get_tk_widget().pack()
    #
    # ax = fig.add_subplot()
    # im = ax.imshow(data)
    #

    #
    # tkinter.mainloop()

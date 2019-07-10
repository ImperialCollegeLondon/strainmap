from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from enum import Flag, auto
from typing import Dict, Optional, Callable
from time import time
from functools import partial
import weakref


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
    UP = auto()
    DOWN = auto()
    SCROLL = UP | DOWN


class Action(Flag):
    DRAG = auto()
    CLICK = auto()
    SCROLL = auto()


ACTIONS: Dict = dict()
"""Registry of the available mouse actions."""


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


def get_action(*keys):
    """Provides the action corresponding to the given keys."""

    def dummy(*args):
        pass

    return ACTIONS.get(keys, dummy)


class PlotInteractions(object):
    @classmethod
    def new_figure(cls, figure_canvas=None, master=None, **kwargs):
        """Factory method to create a new figure with interactive functionality.

        If figure_canvas is provided, this must be a subclass of FigureCanvasBase
        (e.g. FigureCanvasQTAgg, FigureCanvasTkAgg, FigureCanvasKivyAgg, etc.). For
        FigureCanvasTkAgg, then the parent widget need to be provided, too. To show the
        plot, the canvas in figure.canvas will need to be incorporated to the rest
        of the frontend. E.g:

        Tkinter:
            figure.canvas.get_tk_widget().pack()
        PyQt:
            parent_widget.addWidget(figure.canvas)

        If figure_canvas is not provided, then pyplot is used to create a plot using
        the systems default window manager. To show the plot, a call to pyplot.show()
        will do.
        """

        if figure_canvas is None:
            figure = plt.figure(**kwargs)
        else:
            figure = Figure(**kwargs)
            if master is not None:
                figure_canvas(figure, master)
            else:
                figure_canvas(figure)

        return cls.from_figure(figure, **kwargs)

    @classmethod
    def from_figure(cls, figure: Figure, **kwargs):
        """Adds interactions to an existing figure."""
        figure.interactions = cls(figure, **kwargs)
        return figure

    def __init__(
        self, figure: Figure, axis_fraction=0.2, delta_time=0.2, step_factor=1.1
    ):

        self._figure = weakref.ref(figure)
        self.axis_fraction = axis_fraction
        self.step_factor = step_factor
        self.delta_time = delta_time
        self._time_init = 0
        self._event = None
        self._action = None

        # Connect the relevant events
        self.figure.canvas.mpl_connect("button_press_event", self._on_mouse_clicked)
        self.figure.canvas.mpl_connect("button_release_event", self._on_mouse_released)
        self.move_event_id = self.figure.canvas.mpl_connect(
            "motion_notify_event", self._on_mouse_moved
        )
        self.figure.canvas.mpl_connect("scroll_event", self._on_scrolled)

    @property
    def figure(self):
        """The Figure this interaction is connected to."""
        return self._figure()

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
        if time() - self.time_init > self.delta_time:
            self._event = None
            self._action = None
            return

        location = self._get_mouse_location(self._event.xdata, self._event.ydata)
        button = self._get_mouse_button(self._event.button)
        action = Action.CLICK

        get_action(location, button, action)(self, self._event)

        self._event = None
        self._action = None

    def _on_mouse_moved(self, event):
        """Runs the action due to dragging if time since click is too long."""
        if self._event is None or event.inaxes is None:
            return
        elif time() - self.time_init < self.delta_time:
            return
        elif self._action is None:
            location = self._get_mouse_location(self._event.xdata, self._event.ydata)
            button = self._get_mouse_button(self._event.button)
            action = Action.DRAG
            self._action = get_action(location, button, action)

        self._action(self, event)

    def _on_scrolled(self, event):
        pass

    @staticmethod
    def _get_mouse_button(button):
        """Assigns a logical value for the mouse button that was clicked."""
        return {
            1: Button.LEFT,
            2: Button.CENTRE,
            3: Button.RIGHT,
            "up": Button.SCROLL,
            "down": Button.SCROLL,
        }[button]

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
        self.figure.canvas.draw()

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
        self.figure.canvas.draw()

        self._event = event

    @register_action(location=Location.ANY, button=Button.RIGHT, action=Action.DRAG)
    def _brigthness_and_contrast(self, event):
        """Controls the brightness and contrast of an image.

        If there are more than one image in the axes, only the last one is used.
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

        clim = self._event.inaxes.get_images()[-1].properties()["clim"]
        cspan = clim[1] - clim[0]

        # Movement in Y controls the brightness
        clim_low, clim_high = clim[0] + cspan * deltay, clim[1] + cspan * deltay

        # Movement in X controls the contrast
        clim_low, clim_high = clim_low + cspan * deltax, clim_high - cspan * deltax

        self._event.inaxes.get_images()[-1].update({"clim": (clim_low, clim_high)})
        self.figure.canvas.draw()

        self._event = event

    @register_action(location=Location.ANY, button=Button.SCROLL, action=Action.SCROLL)
    def _scroll_frames(self, event):
        print("Scrolling...")

    @register_action(location=Location.SE, button=Button.LEFT, action=Action.CLICK)
    def _edit_mode(self, event):
        print("Entering Edit mode... Is it really needed?")

    @register_action(location=Location.SW, button=Button.LEFT, action=Action.CLICK)
    def _reset_zoom_and_pan(self, event):
        """Resets the axis limits to the original ones, before any zoom and pan."""
        self._event.inaxes.relim()
        self._event.inaxes.autoscale()
        self.figure.canvas.draw()

    @register_action(location=Location.NE, button=Button.LEFT, action=Action.CLICK)
    def _reset_brighness(self, event):
        print("Reset brightness.")

    @register_action(location=Location.NW, button=Button.LEFT, action=Action.CLICK)
    def _reset_contrast(self, event):
        print("Reset contrast.")


if __name__ == "__main__":
    import numpy as np

    data = np.random.random((100, 100))

    # Using Pyplot
    fig = PlotInteractions.new_figure()
    ax = fig.add_subplot()
    im = ax.imshow(data, cmap=plt.get_cmap("binary"))
    ax.get_images()[0].update({"clim": (-1, 0.5)})
    plt.show()

    # In Tkinter
    # import tkinter
    # from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    #
    # fig = PlotInteractions.new_figure(
    #     figure_canvas=FigureCanvasTkAgg, master=tkinter.Tk()
    # )
    # fig.canvas.get_tk_widget().pack()
    #
    # ax = fig.add_subplot()
    # im = ax.imshow(data)
    #

    #
    # tkinter.mainloop()

from .figure_actions_manager import (
    Location,
    Button,
    MouseAction,
    TriggerSignature,
    ActionBase,
)
import numpy as np
from collections import deque
from typing import Callable, Tuple, List, Dict
from functools import partial
from matplotlib.patches import CirclePolygon
from matplotlib.lines import Line2D


def get_deltas(event, last_event):
    pixel_to_data = last_event.inaxes.transData.inverted()
    data = pixel_to_data.transform_point((event.x, event.y))
    last_data = pixel_to_data.transform_point((last_event.x, last_event.y))

    deltax = data[0] - last_data[0]
    deltay = data[1] - last_data[1]

    return deltax, deltay


class ZoomAndPan(ActionBase):
    def __init__(
        self,
        zoom=TriggerSignature(Location.EDGE, Button.LEFT, MouseAction.DRAG),
        pan=TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.DRAG),
        reset_zoom_and_pan=TriggerSignature(
            Location.SW, Button.LEFT, MouseAction.DCLICK
        ),
    ):

        super().__init__()

        self._signatures = {
            zoom: self.zoom,
            pan: self.pan,
            reset_zoom_and_pan: self.reset_zoom_and_pan,
        }

    def zoom(self, event, last_event):
        """Implements the zoom in and out functionality."""
        if not event.inaxes:
            return last_event

        deltax, deltay = get_deltas(event, last_event)

        # Find relative displacement in x
        xlim = last_event.inaxes.get_xlim()
        xspan = xlim[1] - xlim[0]
        deltax = deltax / xspan

        # Find relative displacement in y
        ylim = last_event.inaxes.get_ylim()
        yspan = ylim[1] - ylim[0]
        deltay = deltay / yspan

        delta = np.sign(deltax + deltay) * np.sqrt(deltax ** 2 + deltay ** 2)

        # Update axis limits
        new_xlim = (xlim[0] - delta * xspan, xlim[1] + delta * xspan)
        new_ylim = (ylim[0] - delta * yspan, ylim[1] + delta * yspan)

        last_event.inaxes.set_xlim(new_xlim)
        last_event.inaxes.set_ylim(new_ylim)

        return event

    def pan(self, event, last_event):
        """Pan functionality."""
        if not event.inaxes:
            return last_event

        deltax, deltay = get_deltas(event, last_event)

        # Pan in x direction
        xlim = last_event.inaxes.get_xlim()
        ylim = last_event.inaxes.get_ylim()

        new_xlim = xlim[0] - deltax, xlim[1] - deltax
        new_ylim = ylim[0] - deltay, ylim[1] - deltay

        last_event.inaxes.set_xlim(new_xlim)
        last_event.inaxes.set_ylim(new_ylim)

        return event

    def reset_zoom_and_pan(self, event, *args):
        """Resets the axis limits to the original ones, before any zoom and pan."""
        event.inaxes.relim()
        event.inaxes.autoscale()


class BrightnessAndContrast(ActionBase):
    def __init__(
        self,
        brigthness_and_contrast=TriggerSignature(
            Location.ANY, Button.RIGHT, MouseAction.DRAG
        ),
        reset_brighness_and_contrast=TriggerSignature(
            Location.NW, Button.LEFT, MouseAction.DCLICK
        ),
    ):

        super().__init__()

        self._signatures = {
            brigthness_and_contrast: self.brigthness_and_contrast,
            reset_brighness_and_contrast: self.reset_brighness_and_contrast,
        }

    def brigthness_and_contrast(self, event, last_event):
        """Controls the brightness and contrast of an image.

        If there are more than one image in the axes, the one on top (the last
        one on
        the list) is used.
        """
        if len(last_event.inaxes.get_images()) == 0:
            return

        deltax, deltay = get_deltas(event, last_event)

        # Find relative displacement in x
        xlim = last_event.inaxes.get_xlim()
        xspan = xlim[1] - xlim[0]
        deltax = deltax / xspan

        # Find relative displacement in y
        ylim = last_event.inaxes.get_ylim()
        yspan = ylim[1] - ylim[0]
        deltay = deltay / yspan

        clim = last_event.inaxes.get_images()[-1].get_clim()
        cspan = clim[1] - clim[0]

        # Movement in Y controls the brightness
        clim_low, clim_high = clim[0] + cspan * deltay, clim[1] + cspan * deltay

        # Movement in X controls the contrast
        clim_low, clim_high = clim_low + cspan * deltax, clim_high - cspan * deltax

        last_event.inaxes.get_images()[-1].set_clim(clim_low, clim_high)

        return event

    def reset_brighness_and_contrast(self, event, *args):
        """Resets the brightness and contrast to the limits based on the data."""
        array = event.inaxes.get_images()[-1].get_array()
        clim_low, clim_high = array.min(), array.max()
        event.inaxes.get_images()[-1].set_clim(clim_low, clim_high)


class ScrollFrames(ActionBase):
    def __init__(
        self,
        scroll_frames=TriggerSignature(Location.ANY, Button.CENTRE, MouseAction.SCROLL),
        show_frame_number=TriggerSignature(
            Location.ANY, Button.NONE, MouseAction.ENTERAXES
        ),
        hide_frame_number=TriggerSignature(
            Location.ANY, Button.NONE, MouseAction.LEAVEAXES
        ),
    ):
        super().__init__()
        self._img_shift = 0
        self._signatures = {
            scroll_frames: self.scroll_frames,
            show_frame_number: self.show_frame_number,
            hide_frame_number: self.hide_frame_number,
        }

    def scroll_frames(self, event, *args):
        """The images available in the axes, if more than one, are scrolled."""
        img = deque(event.inaxes.images)
        if len(img) <= 1:
            return

        self._img_shift += event.step
        self._img_shift = self._img_shift % len(img)
        event.inaxes.set_title(f"Frame: {self._img_shift}", loc="left")
        img.rotate(event.step)
        event.inaxes.images = list(img)

    def show_frame_number(self, event, *args):
        """Shows the frame number on top of the axes."""
        if len(event.inaxes.images) > 1:
            event.inaxes.set_title(f"Frame: {self._img_shift}", loc="left")

    @staticmethod
    def hide_frame_number(event, *args):
        """Hides the frame number on top of the axes."""
        event.inaxes.set_title("", loc="left")


def simple_close_contour(points: List[Tuple[float, float]], **kwargs) -> np.ndarray:
    """Adds the first point to the end of the list and returns the resulting array."""
    return np.array(points + [points[0]]).T


def circle(points: List[Tuple[float, float]], resolution=360) -> np.ndarray:
    """Calculates the points of the perimeter of a circle."""
    radius = np.linalg.norm(np.array(points[0] - np.array(points[1])))
    circle = CirclePolygon(points[0], radius, resolution=resolution)
    verts = circle.get_path().vertices
    trans = circle.get_patch_transform()

    return trans.transform(verts).T


class DrawSegments(ActionBase):
    def __init__(
        self,
        num_points: int = -1,
        callback: Callable = simple_close_contour,
        add_point=TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.CLICK),
        remove_point=TriggerSignature(Location.CENTRE, Button.RIGHT, MouseAction.PICK),
        clear_drawing=TriggerSignature(
            Location.CENTRE, Button.RIGHT, MouseAction.DCLICK
        ),
        **kwargs,
    ):
        super().__init__()
        self.num_points = num_points
        self.callback = partial(callback, **kwargs)
        self.points: List = []
        self.marks: List = []
        self.contour: Dict = {}
        self._signatures = {
            add_point: self.add_point,
            remove_point: self.remove_point,
            clear_drawing: self.clear_drawing,
        }

    def add_point(self, _, event, *args):
        """Records the position of the click and marks it on the plot."""
        if len(self.points) == self.num_points:
            return

        self.points.append((event.xdata, event.ydata))

        line = Line2D([event.xdata], [event.ydata], marker="o", color="r", picker=5)
        event.inaxes.add_line(line)
        self.marks.append(line)

        if len(self.points) == self.num_points:
            self.plot_segment(event.inaxes)

    def remove_point(self, _, event, *args):
        """Removes from the list the point that has been clicked."""
        ids = [id(m) for m in self.marks]
        if len(self.points) == 0 or id(event.artist) not in ids:
            return

        index = ids.index(id(event.artist))
        self.points.pop(index)
        self.marks.pop(index).remove()
        if self.contour.get(event.mouseevent.inaxes, None) is not None:
            self.contour.pop(event.mouseevent.inaxes).remove()

    def clear_drawing(self, event, *args):
        """Clears all the data accumulated in the drawing."""
        self.points = []
        for mark in self.marks:
            mark.remove()
        self.marks = []
        if self.contour.get(event.inaxes, None) is not None:
            self.contour.pop(event.inaxes).remove()

    def plot_segment(self, axes):
        """Plots the data resulting from the callback.

        Normally, this will be a figure constructed out of the points clicked in the
        figure, for example a circle, an spline, etc."""
        data = self.callback(self.points)
        self.contour[axes] = Line2D(data[0], data[1], color="r")
        axes.add_line(self.contour[axes])

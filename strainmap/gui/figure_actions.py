from .figure_actions_manager import (
    Location,
    Button,
    MouseAction,
    TriggerSignature,
    ActionBase,
)
import numpy as np
from scipy import interpolate
from typing import Callable, Dict, Optional
from functools import partial
from matplotlib.patches import CirclePolygon
from matplotlib.lines import Line2D
from collections import defaultdict


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

        super().__init__(
            signatures={
                zoom: self.zoom,
                pan: self.pan,
                reset_zoom_and_pan: self.reset_zoom_and_pan,
            }
        )

    @staticmethod
    def zoom(event, last_event):
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

    @staticmethod
    def pan(event, last_event):
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

    @staticmethod
    def reset_zoom_and_pan(event, *args):
        """Resets the axis limits to the original ones, before any zoom and pan."""
        event.inaxes.relim()
        event.inaxes.autoscale()


class BrightnessAndContrast(ActionBase):
    def __init__(
        self,
        brightness_and_contrast=TriggerSignature(
            Location.ANY, Button.RIGHT, MouseAction.DRAG
        ),
        reset_brighness_and_contrast=TriggerSignature(
            Location.NW, Button.LEFT, MouseAction.DCLICK
        ),
    ):

        super().__init__(
            signatures={
                brightness_and_contrast: self.brightness_and_contrast,
                reset_brighness_and_contrast: self.reset_brightness_and_contrast,
            }
        )

    @staticmethod
    def brightness_and_contrast(event, last_event):
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

    @staticmethod
    def reset_brightness_and_contrast(event, *args):
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
        super().__init__(
            signatures={
                scroll_frames: self.scroll_frames,
                show_frame_number: self.show_frame_number,
                hide_frame_number: self.hide_frame_number,
            }
        )
        self._img_shift = 0
        self._images = {}

    def scroll_frames(self, event, *args):
        """The images available in the axes, if more than one, are scrolled."""
        self._img_shift += event.step
        self._img_shift = self._img_shift % len(self._images[event.inaxes])
        event.inaxes.images[0] = self._images[event.inaxes][self._img_shift]
        event.inaxes.set_title(f"Frame: {self._img_shift}", loc="left")

    def show_frame_number(self, event, *args):
        """Shows the frame number on top of the axes.

        If it is the first time, all images but one are extracted from the axes."""
        if self._images.get(event.inaxes, None) is None:
            self.extract_images(event)

        event.inaxes.set_title(f"Frame: {self._img_shift}", loc="left")

    @staticmethod
    def hide_frame_number(event, *args):
        """Hides the frame number on top of the axes."""
        event.inaxes.set_title("", loc="left")

    def extract_images(self, event):
        """Images are extracted from the axes.

        Scrolling with many images loaded in the axes is very expensive, so the first
        time the pointer entyers into the axes, the images are extracted from the axes.
        When scrolling, they will be added back one at a time, replacing the one on
        display."""
        self._images[event.inaxes] = []
        for i in range(len(event.inaxes.images)):
            self._images[event.inaxes].append(event.inaxes.images.pop())

        event.inaxes.images.clear()
        event.inaxes.images = [self._images[event.inaxes][self._img_shift]]


def circle(
    points: np.ndarray, resolution=360, points_per_contour=2, **kwargs
) -> Optional[np.ndarray]:
    """Calculates the points of the perimeter of a circle."""
    if points.shape[1] == 1 or points.shape[0] % points_per_contour == 1:
        return None

    radius = np.linalg.norm(points[-2] - points[-1])
    circle = CirclePolygon(points[-2], radius, resolution=resolution)
    verts = circle.get_path().vertices
    trans = circle.get_patch_transform()

    return trans.transform(verts).T


def simple_closed_contour(
    points: np.ndarray, points_per_contour=6, **kwargs
) -> Optional[np.ndarray]:
    """Adds the first point to the end of the list and returns the resulting array."""
    if points.shape[1] == 1 or points.shape[0] % points_per_contour != 0:
        return None

    data = np.vstack(points[-points_per_contour:], points[-points_per_contour])
    return data.T


def spline(
    points: np.ndarray, points_per_contour=6, resolution=360, degree=3, **kwargs
) -> Optional[np.ndarray]:
    """Returns a spline that passes through the given points."""
    if points.shape[1] == 1 or points.shape[0] % points_per_contour != 0:
        return None

    data = np.vstack((points[-points_per_contour:], points[-points_per_contour]))
    tck, u = interpolate.splprep([data[:, 0], data[:, 1]], s=0, per=True, k=degree)[:2]
    return np.array(interpolate.splev(np.linspace(0, 1, resolution), tck)).T


class DrawContours(ActionBase):
    def __init__(
        self,
        num_contours: int = -1,
        draw_contour: Callable = circle,
        contours_updated: Optional[Callable] = None,
        add_point=TriggerSignature(Location.CENTRE, Button.LEFT, MouseAction.CLICK),
        remove_artist=TriggerSignature(Location.CENTRE, Button.RIGHT, MouseAction.PICK),
        clear_drawing=TriggerSignature(
            Location.CENTRE, Button.RIGHT, MouseAction.DCLICK
        ),
        **kwargs,
    ):
        """Add the capability of drawing contours in a figure.

        This action enables to draw points in an axes and draw a contour out of them.
        The points (and the resulting contours) can be removed. By default, it draws
        circles every 2 points, but the user can provide a draw_contour function that
        uses the available points in a different way.

        After drawing each contour, contours_updated is called, enabling the user to
        retrieve the data. Alternatively, the data can be directly accessed from:

         - figure.actions_manager.DrawContours.contour_data

        which is a dictionary with all the contours per axes.

        Args:
            num_contours: Number of contours to add. Negative number for unlimited.
            draw_contour: Function used to create the contour. This function should take
                as first argument an array with all the points currently in the axes and
                return an array with the data to plot. Kwargs of this call will be
                passed to this function.
            contours_updated: Function called whenever the number of contours changes.
                It should take the list of contours as first argument and a list of all
                the points as a second argument. Kwargs of this call will be
                passed to this function.
            add_point: TriggerSignature for this action.
            remove_artist: TriggerSignature for this action.
            clear_drawing: TriggerSignature for this action.
            **kwargs: Arguments passed to either draw_contour or contours_updated.
        """
        super().__init__(
            signatures={
                add_point: self.add_point,
                remove_artist: self.remove_artist,
                clear_drawing: self.clear_drawing,
            }
        )
        self.num_contours = num_contours
        self.contour_callback = partial(draw_contour, **kwargs)
        self.contours_updated = (
            partial(contours_updated, **kwargs)
            if contours_updated is not None
            else lambda *args: None
        )

        self.num_points = -1
        self.points: Dict = defaultdict(list)
        self.contour_data: Dict = defaultdict(list)
        self.marks: Dict = defaultdict(list)
        self.contours: Dict = defaultdict(list)

    def add_point(self, _, event, *args):
        """Records the position of the click and marks it on the plot.

        Args:
            _: The event associated with the button released (ignored).
            event: The event associated with the button click.
            *args: (ignored)

        Returns:
            None
        """
        if (
            len(self.contours[event.inaxes]) == self.num_contours
            or len(self.points[event.inaxes]) == self.num_points
        ):
            return

        self.points[event.inaxes].append((event.xdata, event.ydata))

        line = Line2D([event.xdata], [event.ydata], marker="o", color="r", picker=4)
        event.inaxes.add_line(line)
        self.marks[event.inaxes].append(line)
        self.add_contour(event.inaxes)

    def remove_artist(self, _, event, *args) -> None:
        """ Removes an artist (point or contour) from the plot.

        Args:
            _: The event associated with the button released (ignored).
            event: The event associated with the button click.
            *args: (ignored)

        Returns:
            None
        """
        axes = event.mouseevent.inaxes
        if axes not in self.points:
            return

        ids_marks = [id(m) for m in self.marks[axes]]
        ids_contours = [id(m) for m in self.contours[axes]]

        if id(event.artist) in ids_marks:
            index = ids_marks.index(id(event.artist))
            self.points[axes].pop(index)
            self.marks[axes].pop(index).remove()

        elif id(event.artist) in ids_contours:
            index = ids_contours.index(id(event.artist))
            self.contour_data[axes].pop(index)
            self.contours[axes].pop(index).remove()

            self.contours_updated(  # type: ignore
                self.contour_data[axes], np.array(self.points[axes])
            )

    def clear_drawing(self, event, *args) -> None:
        """ Clears all the data accumulated in the drawing and the axes.

        Args:
            event: The event that triggered this action.
            *args: (ignored)

        Returns:
            None
        """
        axes = event.inaxes

        self.points[axes].clear()
        self.contour_data[axes].clear()

        for mark in self.marks[axes]:
            mark.remove()

        for contour in self.contours[axes]:
            contour.remove()

        self.marks[axes].clear()
        self.contours[axes].clear()

        self.contours_updated(  # type: ignore
            self.contour_data[axes], np.array(self.points[axes])
        )

    def add_contour(self, axes) -> None:
        """ Calls the contour callback and add a contour to the axes with the data.

        When completed, if the data is not none, contours_updated callback is called
        with all the contour data and all the points as arguments.

        The first time the contour is plot, the number of total points is also
        calculated.

        Args:
            axes: Axes to add the contour to.

        Returns:
            None
        """
        data = self.contour_callback(np.array(self.points[axes]))

        if data is not None:
            if self.num_points == -1:
                self.num_points = len(self.points[axes]) * self.num_contours

            self.contour_data[axes].append(data)

            line = Line2D(*data, color="r", picker=2)
            axes.add_line(line)
            self.contours[axes].append(line)

            self.contours_updated(  # type: ignore
                self.contour_data[axes], np.array(self.points[axes])
            )

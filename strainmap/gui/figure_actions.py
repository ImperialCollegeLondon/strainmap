from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.animation as animation
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import CirclePolygon
from scipy import interpolate

from .figure_actions_manager import (
    ActionBase,
    Button,
    Location,
    MouseAction,
    TriggerSignature,
)


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

        delta = np.sign(deltax + deltay) * np.sqrt(deltax**2 + deltay**2)

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
        animate=TriggerSignature(Location.NE, Button.LEFT, MouseAction.DCLICK),
    ):
        super().__init__(
            signatures={scroll_frames: self.scroll_axes, animate: self.animate}
        )
        self._anim = {}
        self._anim_running = {}
        self._current_frames = defaultdict(lambda: 0)
        self._scroller = {}
        self._linked_axes = {}

    def set_scroller(self, scroller, axes):
        """Sets the generator function that will produce the new data when scrolling."""
        self._scroller[axes] = scroller
        axes.set_title("Frame: 0", loc="left")

    def link_axes(self, axes1, axes2):
        """Links two axes, so scrolling happens simultaneously in both."""
        if axes1 in self._linked_axes or axes2 in self._linked_axes:
            raise RuntimeError("An axes can only be linked to a single other.")

        self._linked_axes[axes1] = axes2
        self._linked_axes[axes2] = axes1
        axes1.set_title("Frame: 0", loc="left")
        axes2.set_title("Frame: 0", loc="left")

    def unlink_axes(self, axes1, axes2):
        """Unlinks two linked axes."""
        self._linked_axes.pop(axes1, None)
        self._linked_axes.pop(axes2, None)

    def scroll_axes(self, event, *args):
        """The images available in the axes, if more than one, are scrolled."""
        self._scroll_axes(None, event.step, event.inaxes)

    def animate(self, event, *args):
        """Animate the sequence of images in the axes."""

        fig = event.canvas.figure
        axes = event.inaxes

        if axes not in self._anim:
            self._anim[axes] = animation.FuncAnimation(
                fig, self._scroll_axes, interval=20, fargs=(1, axes)
            )
            self._anim_running[axes] = True

        elif not self._anim_running[axes]:
            self._anim[axes].event_source.start()
            self._anim_running[axes] = True

        else:
            self._anim[axes].event_source.stop()
            self._anim_running[axes] = False

    def stop_animation(self):
        """Stops an animation, if there is one running."""
        for axes in self._anim_running:
            if self._anim_running[axes]:
                self._anim[axes].event_source.stop()

    def _scroll_axes(self, _, step, axes):
        """Internal function that decides what to scroll."""
        step = int(np.sign(step))
        frame = self._current_frames[axes] + step
        self.go_to_frame(frame, axes)

    def go_to_frame(self, frame, axes):
        """Scroll directly to this specific frame in the chosen axes."""
        self._current_frames[axes] = frame
        self.scroll_artists(axes)

        if axes in self._linked_axes:
            self._current_frames[self._linked_axes[axes]] = frame
            self.scroll_artists(self._linked_axes[axes])

    def scroll_artists(self, axes):
        """Actually scroll the data of a single axes."""
        if axes not in self._scroller:
            return

        self._current_frames[axes], img, lines = self._scroller[axes](
            self._current_frames[axes]
        )
        if img is not None:
            axes.images[0].set_data(img)

        if lines is None:
            pass
        elif isinstance(lines, tuple):
            for i, l in enumerate(lines):
                if l is None:
                    continue
                axes.lines[i].set_data(l)
        else:
            axes.lines[0].set_data(lines)

        axes.set_title(f"Frame: {self._current_frames[axes]}", loc="left")

    def clear(self):
        """Removes the information stored in the ScrollFrame object."""
        for ax, anim in self._anim.items():
            anim.event_source.stop()
            del anim
        self._anim = {}
        self._anim_running = {}


def circle(points: np.ndarray, resolution=360, **kwargs) -> Optional[np.ndarray]:
    """Calculates the points of the perimeter of a circle."""
    if points.shape[1] == 1 or points.shape[0] % 2 == 1:
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

    data = np.vstack((points[-points_per_contour:], points[-points_per_contour]))
    return data.T


def spline(
    points: np.ndarray, points_per_contour=6, resolution=360, degree=3, **kwargs
) -> Optional[np.ndarray]:
    """Returns a spline that passes through the given points."""
    if points.shape[1] == 1 or points.shape[0] % points_per_contour != 0:
        return None

    data = np.vstack((points[-points_per_contour:], points[-points_per_contour]))
    tck, u = interpolate.splprep([data[:, 0], data[:, 1]], s=0, per=True, k=degree)[:2]
    result = np.array(interpolate.splev(np.linspace(0, 1, resolution), tck)).T
    return result


def single_point(points: np.ndarray) -> Optional[np.ndarray]:
    """Returns a single point to be plotted 'as as'."""
    return points[-1][np.newaxis].T


class DrawContours(ActionBase):
    def __init__(
        self,
        num_contours: int = -1,
        draw_contour: Callable = circle,
        contours_updated: Optional[Callable] = None,
        add_point=TriggerSignature(Location.CROSS, Button.LEFT, MouseAction.CLICK),
        remove_artist=TriggerSignature(Location.CROSS, Button.RIGHT, MouseAction.PICK),
        clear_drawing=TriggerSignature(
            Location.CROSS, Button.RIGHT, MouseAction.DCLICK
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
                add_point: self._add_point,
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

    def _add_point(self, _, event, *args):
        """Records the position of the click and marks it on the plot.

        Args:
            _: The event associated with the button released (ignored).
            event: The event associated with the button click.
            *args: (ignored)

        Returns:
            None
        """
        self.add_point(event.inaxes, event.xdata, event.ydata)

    def add_point(self, ax, xdata, ydata):
        """Adds a point to the given axes."""
        if (
            len(self.contours[ax]) == self.num_contours
            or len(self.points[ax]) == self.num_points
        ):
            return

        self.points[ax].append((xdata, ydata))

        line = Line2D([xdata], [ydata], marker="o", color="r", pickradius=4)
        line.set_picker(True)
        ax.add_line(line)
        self.marks[ax].append(line)
        self.add_contour(ax)

    def remove_artist(self, _, event, *args) -> None:
        """Removes an artist (point or contour) from the plot.

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
        """Clears all the data accumulated in the drawing and the axes.

        Args:
            event: The event that triggered this action.
            *args: (ignored)

        Returns:
            None
        """
        self.clear_drawing_(event.inaxes)

    def clear_drawing_(self, axes, *args) -> None:
        """Clears all the data accumulated in the drawing and the axes.

        Args:
            axes: The axes from which to delete everything.
            *args: (ignored)

        Returns:
            None
        """
        self.points[axes].clear()
        self.contour_data[axes].clear()

        for mark in self.marks[axes]:
            mark.remove()

        for contour in self.contours[axes]:
            contour.remove()

        self.marks[axes].clear()
        self.contours[axes].clear()

    def add_contour(self, axes) -> None:
        """Calls the contour callback and add a contour to the axes with the data.

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

            line = Line2D(*data, color="r", pickradius=2)
            line.set_picker(True)
            axes.add_line(line)
            self.contours[axes].append(line)

            self.contours_updated(  # type: ignore
                self.contour_data[axes], np.array(self.points[axes])
            )


class DragContours(ActionBase):
    def __init__(
        self,
        contour_fraction: float = 0.15,
        contour_updated: Optional[Callable] = None,
        drag_point=TriggerSignature(Location.ANY, Button.LEFT, MouseAction.PICKDRAG),
        **kwargs,
    ):
        """Add the capability of dragging and deforming closed contours in a figure.

        After updating the shape of a contour, the contour_updated callback is called
        with the new contour data.

        Args:
            contour_fraction: Fraction of the contour length to be used as the width
                of the gaussian that calculates the shifts.
            contour_updated: Function called whenever the number of contours changes.
                It should take the list of contours as first argument and a list of all
                the points as a second argument. Kwargs of this call will be
                passed to this function.
            drag_point: TriggerSignature for this action.
            **kwargs: Arguments passed to either draw_contour or contours_updated.
        """

        super().__init__(signatures={drag_point: self.drag_point})
        self.contour_fraction = np.clip(contour_fraction, a_min=0, a_max=1)
        self.disabled = False
        self._current_artist = None
        self._ignore_drag: List[Line2D] = []
        self._drag_handle = 0
        self._contour_updated = (
            partial(contour_updated, **kwargs)
            if contour_updated is not None
            else lambda *args: None
        )

    def set_contour_updated(self, contour_updated: Callable):
        """Sets the function to be called when the contour is updated."""
        self._contour_updated = contour_updated

    def ignore_dragging(self, artist):
        """Dragging will be ignored for this pickable artist."""
        self._ignore_drag.append(artist)

    def drag_point(self, event, last_event, *args):
        """Drags a point and all the neighbouring ones of a closed contour."""
        if self.disabled:
            return

        if hasattr(last_event, "artist"):
            if last_event.artist in self._ignore_drag:
                self._current_artist = None
            elif isinstance(last_event.artist, Line2D):
                self._current_artist = last_event.artist
                xdata = self._current_artist.get_xdata()
                ydata = self._current_artist.get_ydata()

                xdif = xdata - last_event.mouseevent.xdata
                ydif = ydata - last_event.mouseevent.ydata

                self._drag_handle = (xdif**2 + ydif**2).argmin()

        if self._current_artist is None:
            return

        xdata = self._current_artist.get_xdata()
        ydata = self._current_artist.get_ydata()

        ev = last_event.mouseevent if hasattr(last_event, "mouseevent") else last_event

        deltax, deltay = get_deltas(event, ev)
        shiftx, shifty = self.calculate_shifts(xdata, ydata, deltax, deltay)

        newx = xdata + shiftx
        newy = ydata + shifty

        self._current_artist.set_xdata(newx)
        self._current_artist.set_ydata(newy)

        self._contour_updated(
            self._current_artist.get_label(),
            self._current_artist.axes,
            np.array([newx, newy]),
        )

        return event

    def calculate_shifts(
        self, xdata: np.ndarray, ydata: np.ndarray, deltax: float, deltay: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the shifts for each contour point based on its distance to handle.

        The distance of each point to the handle is calculated when going along the
        contour in both directions and then getting the one that is smaller in each
        case. This distance is used to calculate the displacement for each point based
        on a gaussian curve centered at the handle and with a width defined as a
        fraction of the total length of the contour.

        To avoid the closed contour to "open" due to the lost of accuracy, the
        difference between both ends of the arrays, supposed equal, is forced to be
        mathematically zero.

        Args:
            xdata: Array with the x coordinates of the contour.
            ydata: Array with the y coordinates of the contour.
            deltax: X displacement of the handle.
            deltay: Y displacement of the handle.

        Returns:
            A tuple with the calculated displacement of each point in the segment.
        """
        segment = np.array([xdata, ydata])

        diff = np.linalg.norm(segment - np.roll(segment, -1, axis=1), axis=0)
        diff[0] = 0

        posdir = np.roll(
            np.cumsum(np.roll(diff, -self._drag_handle)), self._drag_handle
        )
        negdir = np.roll(
            np.cumsum(np.roll(diff, -self._drag_handle - 1)[::-1])[::-1],
            self._drag_handle + 1,
        )
        distance = np.minimum(posdir, negdir)
        distance -= distance[self._drag_handle]

        sigma = posdir.max() * self.contour_fraction
        shift = np.exp(-((distance / sigma) ** 2))

        return deltax * shift, deltay * shift


class Markers(ActionBase):
    def __init__(
        self,
        marker_moved: Optional[Callable] = None,
        drag_marker=TriggerSignature(Location.ANY, Button.LEFT, MouseAction.PICKDRAG),
        move_finish=TriggerSignature(Location.ANY, Button.LEFT, MouseAction.RELEASE),
    ):
        """Add sliding markers to read the data from a plot.

        Args:
            marker_moved: Function called whenever a marker is dragged.
            drag_marker: TriggerSignature for this action.
        """

        super().__init__(
            signatures={drag_marker: self.drag_marker, move_finish: self.move_finish}
        )
        self.disabled = False
        self._current_marker = None
        self._current_data = None
        self._linked_data: Dict[Line2D, Union[Line2D, None]] = dict()
        self._marker_moved = (
            marker_moved if marker_moved is not None else lambda *args: None
        )

    def set_marker_moved(self, marker_moved: Callable):
        """Sets the function to be called when the contour is updated."""
        self._marker_moved = marker_moved

    def add_marker(self, line=None, axes=None, xy=None, vline=False, **kwargs):
        """Adds a marker to the axis of the linked data."""
        if line is not None:
            axes = line.axes
            x, y = line.get_data()
            x, y = x[0], y[0]
        elif axes is not None:
            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            x, y = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
        else:
            raise ValueError("At least one of 'lines' or 'axes' must be defined.")

        if xy is not None:
            x, y = xy[0], xy[1]

        if vline:
            options = dict(pickradius=12, linewidth=3, linestyle="--")
            options.update(kwargs)
            marker = axes.axvline(x, **options)
            marker.set_picker(True)
        else:
            options = dict(pickradius=12, marker="x", markersize=20, linestyle="None")
            options.update(kwargs)
            marker = axes.plot(x, y, **options)[0]
            marker.set_picker(True)

        self._linked_data[marker] = line

        return marker

    def update_marker_position(self, marker, new_x, new_y=None):
        """Updates the position of an existing marker."""
        line = self._linked_data.get(marker, "Not a marker")

        if line == "Not a marker":
            return
        elif line is None:
            y = marker.get_ydata()[0]
            marker.set_data([new_x], [new_y if new_y is not None else y])
        else:
            x, y, idx = self.get_closest(line, new_x)
            old_x = marker.get_xdata()
            if len(old_x) == 2:
                marker.set_xdata([x, x])
            if len(old_x) == 1:
                marker.set_data([x], [y])

    def drag_marker(self, event, last_event, *args):
        """Drags a marker to a new position of the data."""
        if self.disabled:
            return

        if hasattr(last_event, "artist"):
            if last_event.artist in self._linked_data:
                self._current_marker = last_event.artist
                self._current_data = self._linked_data[self._current_marker]
            else:
                self._current_marker = None
                self._current_data = None

        if self._current_marker is None:
            return

        ev = event.mouseevent if hasattr(event, "mouseevent") else event
        old_x = self._current_marker.get_xdata()

        if self._current_data is None:
            if ev.xdata != old_x:
                self._current_marker.set_data([ev.xdata], [ev.ydata])
        else:
            x, y, idx = self.get_closest(self._current_data, ev.xdata)
            if len(old_x) == 1 and x != old_x[0]:
                self._current_marker.set_data([x], [y])
            elif len(old_x) == 2 and x != old_x[0]:
                self._current_marker.set_xdata([x, x])

        return event

    def move_finish(self, event, last_event, *args):
        """Executes marker move after mouse release."""
        if self._current_marker is None:
            return

        if self._current_data is None:
            x, y = event.xdata, event.ydata
            idx = 0
        else:
            x, y, idx = self.get_closest(self._current_data, event.xdata)

        self._marker_moved(self._current_marker, self._current_data, x, y, idx)
        self._current_marker = None

    @staticmethod
    def get_closest(line, mx):
        x, y = line.get_data()
        mini = np.abs(x - mx).argmin().item()
        return x[mini], y[mini], mini


class SimpleScroller(ActionBase):
    """Simpler scroller that links the scroll functionality to an external function."""

    def __init__(
        self, scroll=TriggerSignature(Location.ANY, Button.CENTRE, MouseAction.SCROLL)
    ):
        super().__init__(signatures={scroll: self.scroller})
        self._scroller = None
        self.disabled = False

    def set_scroller(self, scroller, *args, **kwargs):
        """The function to be called when scrolling."""
        self._scroller = partial(scroller, *args, **kwargs)

    def scroller(self, event, *args):
        if not self.disabled:
            self._scroller(int(np.sign(event.step)))

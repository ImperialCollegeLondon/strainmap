import tkinter as tk
from itertools import chain, product
from tkinter import ttk
from typing import Callable, Dict, Tuple

import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from strainmap.coordinates import Comp, Mark, Region
from strainmap.gui.figure_actions import Markers, SimpleScroller
from strainmap.gui.figure_actions_manager import FigureActionsManager


class MarkersFigure:

    components: Tuple[Comp, ...] = (Comp.LONG, Comp.RAD, Comp.CIRC)

    def __init__(
        self,
        values: xr.DataArray,
        cylindrical: xr.DataArray,
        images: xr.DataArray,
        markers: xr.DataArray,
        master: ttk.Frame,
        xlabel: str = "Frame",
        ylabel: str = "Velocity (cm/s)",
        colours: Tuple[str, ...] = ("red", "green", "blue") * 2
        + ("orange", "darkblue", "purple"),
        ids: Tuple[Comp, Mark] = tuple(
            chain(
                product((Comp.LONG, Comp.RAD), (Mark.PS, Mark.PD, Mark.PAS)),
                product((Comp.CIRC,), (Mark.PC1, Mark.PC2, Mark.PC3)),
            )
        ),
    ):
        self.ids = ids
        self.limits: Tuple[int, ...] = ()

        self.fig = Figure(constrained_layout=True)
        canvas = FigureCanvasTkAgg(self.fig, master=master)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
        toolbar_frame = ttk.Frame(master=master)
        toolbar_frame.grid(row=1, column=0, sticky=tk.NSEW)
        NavigationToolbar2Tk(canvas, toolbar_frame)

        self.fig.actions_manager = FigureActionsManager(
            self.fig, Markers, SimpleScroller
        )
        gs = self.fig.add_gridspec(2, 9, height_ratios=[5, 2])

        self.axes = self.add_line_subplots(gs, xlabel, ylabel)
        self.maps = self.add_image_subplots(gs, colours)
        self.lines = self.add_lines(values)
        self.images, self.cylindrical, self.cbar = self.images_and_cylindrical_data(
            images, cylindrical, markers
        )
        self.marker_artists, self.fixed_markers = self.add_markers(markers, colours)

        self.draw()

    def draw(self) -> None:
        """Shortcut to re-draw the figure"""
        self.fig.canvas.draw_idle()

    def set_marker_moved(self, callback: Callable) -> None:
        """Sets the function to be called when one marker is moved

        Args:
            callback (Callable): The function to be called.

        Returns:
            None
        """
        self.fig.actions_manager.Markers.set_marker_moved(callback)  # type: ignore

    def set_scroller(self, callback, Callable) -> None:
        """Sets the function to be called when scrolling over the plots

        Args:
            callback (Callable): The function to be called.

        Returns:
            None
        """
        self.fig.actions_manager.SimpleScroller.set_scroller(callback)  # type: ignore

    def add_line_subplots(
        self, gs: GridSpec, xlabel: str, ylabel: str
    ) -> Dict[Comp, Axes]:
        """Add the line subplots.

        Args:
            gs (GridSpec): Specification of the grid for the axes.
            xlabel (str): Label for the x axes
            ylabel (str): Label for the y axes

        Returns:
            Dictionary with the subplots for each of the components.
        """
        output: Dict[Comp, Axes] = {}
        for i, comp in self.components:
            output[comp] = self.fig.add_subplot(gs[0, 3 * i : 3 * (i + 1)])
            output[comp].axhline(color="k", lw=1)
            output[comp].set_title(Comp.value)
            output[comp].set_xlabel(xlabel)

        output[Comp.LONG].set_ylabel(ylabel)

        return output

    def add_lines(self, values: xr.DataArray) -> Dict[Comp, Line2D]:
        """Add lines to the line plots.

        Args:
            values (xr.DataArray): Array with the data to plot. Should have two
                dimensions, 'comp' and 'frame', and the former should have three
                coordinates, Comp.LONG, Comp.RAD and Comp.CIRC.

        Returns:
            Dictionary with the lines for each of the components.
        """
        output: Dict[Comp, Line2D] = {}
        for i, comp in self.components:
            output[comp] = self.axes[comp].plot(
                values.frame, values.sel(comp=comp), "k", label=comp.name
            )[0]
            self.axes[comp].relim()
            self.axes[comp].autoscale(False)
        return output

    def add_image_subplots(
        self, gs: GridSpec, colours: Tuple[str, ...]
    ) -> Dict[Tuple[Comp, Mark], Axes]:
        """Add the image subplots.

        Args:
            gs (GridSpec): Specification of the grid for the axes.
            colours (Tuple[str, ...]): Colours for the frames of the subplots

        Returns:
            Dictionary with the subplots for each of the components.
        """
        output: Dict[Tuple[Comp, Mark], Axes] = {}

        for i, loc in enumerate(self.ids):
            output[loc] = self.fig.add_subplot(gs[1, i])
            output[loc].get_xaxis().set_visible(False)
            output[loc].get_yaxis().set_visible(False)
            for side in ["left", "right", "bottom", "top"]:
                output[loc].spines[side].set_color(colours[i])
                output[loc].spines[side].set_linewidth(3)

        return output

    def images_and_cylindrical_data(
        self, images: xr.DataArray, cylindrical: xr.DataArray, markers: xr.DataArray
    ) -> Tuple[
        Dict[Tuple[Comp, Mark], AxesImage], Dict[Tuple[Comp, Mark], AxesImage], Colorbar
    ]:
        """Add bg and masks to the map subplots.

        Args:
            images (xr.DataArray): Background images for the color plots; typically the
                magnitude images.
            cylindrical (xr.DataArray): Velocity field in cylindrical coordinates, to
                overlay on top of the images.
            markers (xr.DataArray): Markers information.

        Returns:
            Tuple with the images plots, the cylindrical data overlay and the colorbar.
        """
        img: Dict[Tuple[Comp, Mark], AxesImage] = {}
        cyl: Dict[Tuple[Comp, Mark], AxesImage] = {}

        if Region.GLOBAL in set(cylindrical.region.data):
            self.limits = self.find_limits(cylindrical)

        rmin, rmax, cmin, cmax = self.limits
        vmin, vmax = cylindrical.min(), cylindrical.max()
        for i, (comp, m) in enumerate(self.ids):
            frame = int(markers.sel(marker=m, comp=comp, quantity="frame").item())

            # Background magnitude images
            img[(comp, m)] = self.maps[(comp, m)].imshow(
                images.sel(
                    frame=frame, row=slice(rmin, rmax + 1), col=slice(cmin, cmax + 1)
                ),
                cmap=plt.get_cmap("binary_r"),
            )

            # Overlay cylindrical velocity images
            cyl[(comp, m)] = self.maps[(comp, m)].imshow(
                cylindrical.sel(
                    comp=comp,
                    frame=frame,
                    row=slice(rmin, rmax + 1),
                    col=slice(cmin, cmax + 1),
                ),
                cmap=plt.get_cmap("seismic"),
                vmin=vmin,
                vmax=vmax,
            )

        cbar = self.fig.colorbar(cyl[self.ids[0]], ax=self.maps[self.ids[0]], pad=-1.4)

        return img, cyl, cbar

    @staticmethod
    def find_limits(mask: xr.DataArray, margin: int = 20) -> Tuple[int, ...]:
        """Find the appropriate limits of a masked array in order to plot it nicely."""
        cmin = mask.min(dim="col") - margin
        rmin = mask.min(dim="row") - margin
        cmax = mask.max(dim="col") + margin
        rmax = mask.max(dim="row") + margin

        return rmin, rmax, cmin, cmax

    def add_markers(
        self, markers: xr.DataArray, colours: Tuple[str, ...]
    ) -> Tuple[Dict[Tuple[Comp, Mark], Line2D], Dict[Tuple[Comp, Mark], Line2D]]:
        """Adds markers to the plots.

        Args:
            markers (xr.DataArray): Markers information.
            colours (Tuple[str, ...]): Colours for the markers

        Returns:

        """
        add_marker = self.fig.actions_manager.Markers.add_marker
        markers_artists: Dict[Tuple[Comp, Mark], Line2D] = {}
        fixed_markers: Dict[Tuple[Comp, Mark], Line2D] = {}

        for i, (comp, m) in enumerate(self.ids):
            markers_artists[(comp, m)] = add_marker(
                self.lines[comp],
                xy=markers.sel(
                    marker=m, comp=comp, quantity=["frame", "velocity"]
                ).data,
                label=m.text,
                color=colours[i],
                marker=str(i % 3 + 1),
                markeredgewidth=1.5,
                markersize=15,
            )

        # The ES marker is a fixed line in all plots...
        es_marker = markers.sel(
            marker=Mark.ES, comp=Comp.RAD, quantity=["frame", "velocity"]
        ).data
        for comp in (Comp.LONG, Comp.RAD, Comp.CIRC):
            fixed_markers[(comp, Mark.ES)] = self.lines[comp].axes.axvline(
                es_marker[0], color="grey", linewidth=2, linestyle="--"
            )

        # But, additionally in the radial it can be moved.
        markers_artists[(Comp.RAD, Mark.ES)] = add_marker(
            self.lines[Comp.RAD],
            xy=es_marker,
            vline=True,
            label="ES",
            color="black",
            linewidth=2,
        )

        for comp in (Comp.LONG, Comp.RAD, Comp.CIRC):
            self.axes[comp].legend(frameon=False, markerscale=0.7)

        return markers_artists, fixed_markers

    def update_line(self, data: xr.DataArray, draw=False) -> None:
        """Updates the data of the chosen line.

        Args:
            data (xr.DataArray): Array with the new data for the given component.
            draw (bool): If the figure should be re-drawn.

        Returns:
            None
        """
        self.lines[data.comp.item()].set_data([data.frame, data])
        if draw:
            self.draw()

    def update_maps(
        self,
        images: xr.DataArray,
        cylindrical: xr.DataArray,
        marker_frames: xr.DataArray,
        draw: bool = True,
    ):
        """Updates all the maps with new data.

        Args:
            images (xr.DataArray): New background images.
            cylindrical (xr.DataArray): New overlay data.
            marker_frames (xr.DataArray): Markers frame information.
            draw (bool): If the figure should be re-drawn.

        Returns:
            None
        """
        for i, (comp, m) in enumerate(self.ids):
            frame = int(marker_frames.sel(marker=m, comp=comp).item())
            self.update_one_map(
                (comp, m),
                images.sel(comp=comp, frame=frame),
                cylindrical.sel(comp=comp, frame=frame),
            )

        if draw:
            self.draw()

    def update_one_map(
        self,
        label: Tuple[Comp, Mark],
        images: xr.DataArray,
        cylindrical: xr.DataArray,
        draw: bool = False,
    ) -> None:
        """Updates the maps corresponding to a single marker.

        Args:
            label (Tuple[Comp, Mark]): Label of the map to update.
            images (xr.DataArray): New background image - not cropped.
            cylindrical (xr.DataArray): New overlay data - not cropped.
            draw (bool): If the figure should be re-drawn.

        Returns:
            None
        """
        rmin, rmax, cmin, cmax = self.limits
        self.images[label].set_data(
            images.sel(
                row=slice(rmin, rmax + 1),
                col=slice(cmin, cmax + 1),
            )
        )
        self.cylindrical[label].set_data(
            cylindrical.sel(
                row=slice(rmin, rmax + 1),
                col=slice(cmin, cmax + 1),
            )
        )
        if draw:
            self.draw()

    def update_markers(self, markers: xr.DataArray, draw=True) -> None:
        """Updates the position of all markers in a figure."""
        update_position = self.fig.actions_manager.Markers.update_marker_position

        for comp, m in self.ids:
            update_position(
                self.marker_artists[(comp, m)],
                markers.sel(marker=m, comp=comp, quantity="frame").item(),
            )

        es_loc = markers.sel(marker=Mark.ES, comp=Comp.RAD, quantity="frame").item()
        update_position(self.marker_artists[(Comp.RAD, Mark.ES)], es_loc)
        for f in self.fixed_markers.values():
            f.set_xdata([es_loc, es_loc])

        if draw:
            self.draw()

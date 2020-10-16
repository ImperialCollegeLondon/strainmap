import tkinter as tk
from tkinter import ttk
from typing import Callable, Tuple, Dict
from itertools import chain, product

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

from strainmap.gui.figure_actions import Markers, SimpleScroller
from strainmap.gui.figure_actions_manager import FigureActionsManager
from strainmap.coordinates import Comp, VelMark as VM


class MarkersFigure:

    components: Tuple[Comp, ...] = (Comp.LONG, Comp.RAD, Comp.CIRC)
    ids = tuple(
        chain(
            product((Comp.LONG, Comp.RAD), (VM.PS, VM.PD, VM.PAS)),
            product((Comp.CIRC,), (VM.PC1, VM.PC2, VM.PC3)),
        )
    )

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
    ):
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
        self.marker_artists = self.add_markers(markers)

        self.draw()

    def draw(self) -> None:
        """Shortcut to re-draw the figure"""
        self.fig.canvas.draw_idle()

    def set_marker_moved(self, callback: Callable) -> None:
        """ Sets the function to be called when one marker is moved

        Args:
            callback (Callable): The function to be called.

        Returns:
            None
        """
        self.fig.actions_manager.Markers.set_marker_moved(callback)  # type: ignore

    def set_scroller(self, callback, Callable) -> None:
        """ Sets the function to be called when scrolling over the plots

        Args:
            callback (Callable): The function to be called.

        Returns:
            None
        """
        self.fig.actions_manager.SimpleScroller.set_scroller(callback)  # type: ignore

    def add_line_subplots(
        self, gs: GridSpec, xlabel: str, ylabel: str
    ) -> Dict[Comp, Axes]:
        """  Add the line subplots.

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
        """ Add lines to the line plots.

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

    def add_image_subplots(self, gs: GridSpec, colours: Tuple[str, ...]):
        """ Add the image subplots.

        Args:
            gs (GridSpec): Specification of the grid for the axes.
            colours (Tuple[str, ...]): Colours for the frames of the subplots

        Returns:
            Dictionary with the subplots for each of the components.
        """
        output: Dict[Tuple[Comp, VM], Axes] = {}

        for i, loc in enumerate(self.ids):
            output[loc] = self.fig.add_subplot(gs[1, i])
            output[loc].get_xaxis().set_visible(False)
            output[loc].get_yaxis().set_visible(False)
            for side in ["left", "right", "bottom", "top"]:
                output[loc].spines[side].set_color(colours[i])
                output[loc].spines[side].set_linewidth(3)

        return output

    def images_and_cylindrical_data(self, images, cylindrical, markers: xr.DataArray):
        """Add bg and masks to the map subplots."""
        img: Dict[Tuple[Comp, VM], Axes] = {}
        cyl: Dict[Tuple[Comp, VM], Axes] = {}

        # if "global" == self.velocities_var.get():
        #     self.limits = self.find_limits(vel_masks[0, 0])

        rmin, rmax, cmin, cmax = self.limits
        vmin, vmax = cylindrical.min(), cyl.max()
        for i, (comp, m) in enumerate(self.ids):
            frame = int(markers.sel(marker=m, comp=comp, quantity="frame").item())
            img[(comp, m)] = self.maps[(comp, m)].imshow(
                    images.sel(frame=frame, row=slice(rmin, rmax + 1), col=slice(cmin, cmax + 1)),
                    cmap=plt.get_cmap("binary_r"),
                )

            cyl[(comp, m)] = self.maps[(comp, m)].imshow(
                    cylindrical.sel(comp=comp, frame=frame, row=slice(rmin, rmax + 1), col=slice(cmin, cmax + 1)),
                    cmap=plt.get_cmap("seismic"),
                    vmin=vmin,
                    vmax=vmax,
                )
            )

        cbar = self.fig.colorbar(masks["_long"][0], ax=self.maps[0], pad=-1.4)

        return bg, masks, cbar

    @staticmethod
    def find_limits(mask, margin=30):
        """Find the appropiate limits of a masked array in order to plot it nicely."""
        rows, cols = mask.nonzero()

        cmin = max(cols.min() - margin, 0)
        rmin = max(rows.min() - margin, 0)
        cmax = min(cols.max() + margin, mask.shape[0])
        rmax = min(rows.max() + margin, mask.shape[1])

        return rmin, rmax, cmin, cmax

    def add_markers(self, markers):
        """Adds markers to the plots.).

        - markers - Contains all the marker information for all regions and components.
            Their shape is [regions, component (3), marker_id (3 or 4), marker_data (3)]
        """
        add_marker = self.fig.actions_manager.Markers.add_marker

        vel_lbl = ["_long"] * 3 + ["_rad"] * 3 + ["_circ"] * 3
        colors = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]

        markers_artists = []
        for i, label in enumerate(vel_lbl):
            markers_artists.append(
                add_marker(
                    self.lines[label],
                    xy=markers[i // 3, i % 3, :2],
                    label=marker_lbl[i],
                    color=colors[i],
                    marker=str(i % 3 + 1),
                    markeredgewidth=1.5,
                    markersize=15,
                )
            )

        for label in ("_long", "_rad", "_circ"):
            self.fixed_markers.append(
                self.lines[label].axes.axvline(
                    self.es_marker[0], color="grey", linewidth=2, linestyle="--"
                )
            )

        markers_artists.append(
            add_marker(
                self.lines["_rad"],
                xy=self.es_marker,
                vline=True,
                label="ES",
                color="black",
                linewidth=2,
            )
        )

        self.axes["_long"].legend(frameon=False, markerscale=0.7)
        self.axes["_rad"].legend(frameon=False, markerscale=0.7)
        self.axes["_circ"].legend(frameon=False, markerscale=0.7)

        return markers_artists

    def update_marker_position(self, marker_label, data_label, new_x):
        """Convenience method to update the marker position."""
        self.actions_manager.Markers.update_marker_position(
            marker_label, data_label, new_x
        )

    def update_line(self, vel_label, data, draw=False):
        """Updates the data of the chosen line."""
        self.lines[vel_label].set_data(data)
        if draw:
            self.draw()

    def update_bg(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.images[vel_label][idx], vel_label, data, draw)

    def update_mask(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.cylindrical[vel_label][idx], vel_label, data, draw)

    def update_data(self, subplot, vel_label, data, draw=False):
        """Common data updating method."""
        subplot.set_data(data)
        self.axes[vel_label].relim()
        self.axes[vel_label].autoscale()

        if draw:
            self.draw()

    def update_maps(
        self, vel_masks: np.ndarray, images: np.ndarray, markers: np.ndarray, draw=True
    ):
        """Updates the maps (masks and background data)."""
        rmin, rmax, cmin, cmax = self.limits
        for i in range(9):
            axes = self.axes_lbl[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            self.update_mask(
                axes, i % 3, vel_masks[i // 3, frame, rmin : rmax + 1, cmin : cmax + 1]
            )
            self.update_bg(axes, i % 3, images[frame, rmin : rmax + 1, cmin : cmax + 1])

        if draw:
            self.draw()

    def update_one_map(
        self,
        vel_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
        axes: str,
        marker_lbl: str,
    ):
        """Updates the maps correspoinding to a single marker."""
        rmin, rmax, cmin, cmax = self.limits
        component = self.axes_lbl.index(axes)
        idx = self.marker_idx[marker_lbl]
        frame = int(markers[component, idx, 0])
        self.update_mask(
            axes, idx, vel_masks[component, frame, rmin : rmax + 1, cmin : cmax + 1]
        )
        self.update_bg(axes, idx, images[frame, rmin : rmax + 1, cmin : cmax + 1])
        self.axes[axes].set_ylim(*self.vel_lim[axes])
        self.draw()

    def update_velocities(self, vels, draw=True):
        """Updates all velocities."""
        x = np.arange(vels.shape[-1])
        for i, label in enumerate(self.axes_lbl):
            self.update_line(label, (x, vels[i]))
        if draw:
            self.draw()

    def update_markers(self, markers, draw=True):
        """Updates the position of all markers in a figure."""
        update_position = self.fig.actions_manager.Markers.update_marker_position

        for i, artist in enumerate(self.marker_artists[:-1]):
            update_position(artist, int(markers[i // 3, i % 3, 0]))

        update_position(self.marker_artists[-1], self.es_marker[0])

        for f in self.fixed_markers:
            f.set_xdata([self.es_marker[0]] * 2)

        if draw:
            self.draw()

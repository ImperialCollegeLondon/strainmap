import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view, trigger_event
from .figure_actions_manager import FigureActionsManager
from .figure_actions import Markers


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED

    def __init__(self, root):

        super().__init__(root, button_text="Velocities", button_image="speed.gif")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.visualise_frame = None
        self.datasets_box = None
        self.datasets_var = tk.StringVar(value="")
        self.velocities_frame = None
        self.velocities_var = tk.StringVar(value="")
        self.phantom_var = tk.BooleanVar(value=False)
        self.gfig = None
        self.regional_fig = None
        self.color_maps = None

        self.create_controls()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self, width=300)
        control.columnconfigure(0, weight=1)
        self.visualise_frame = ttk.Frame(master=self)
        self.visualise_frame.columnconfigure(0, weight=1)
        self.visualise_frame.rowconfigure(0, weight=1)

        # Dataset frame
        dataset_frame = ttk.Labelframe(control, text="Datasets:")
        dataset_frame.columnconfigure(0, weight=1)
        dataset_frame.rowconfigure(0, weight=1)
        dataset_frame.rowconfigure(1, weight=1)

        self.datasets_box = ttk.Combobox(
            master=dataset_frame,
            textvariable=self.datasets_var,
            values=[],
            state="readonly",
        )
        self.datasets_box.bind("<<ComboboxSelected>>", self.dataset_changed)

        # Velocities frame
        self.velocities_frame = ttk.Labelframe(control, text="Velocities:")
        self.velocities_frame.columnconfigure(0, weight=1)
        self.velocities_frame.columnconfigure(1, weight=1)

        add_velocity = ttk.Button(
            master=self.velocities_frame, text="Add", command=self.add_velocity
        )
        remove_velocity = ttk.Button(
            master=self.velocities_frame, text="Remove", command=self.remove_velocity
        )

        # Background frame
        background_frame = ttk.Labelframe(control, text="Background correction:")
        background_frame.columnconfigure(0, weight=1)
        background_frame.rowconfigure(0, weight=1)
        background_frame.rowconfigure(1, weight=1)

        for i, method in enumerate(("Average", "Phantom")):
            ttk.Radiobutton(
                background_frame, text=method, variable=self.phantom_var, value=i
            ).grid(row=i, column=0, sticky=(tk.S, tk.W))

        # Grid all the widgets
        control.grid(row=0, column=0, sticky=tk.NSEW, padx=10, pady=10)
        self.visualise_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=10, pady=10)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.velocities_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        add_velocity.grid(row=99, column=0, sticky=tk.NSEW)
        remove_velocity.grid(row=99, column=1, sticky=tk.NSEW)
        background_frame.grid(row=2, column=0, sticky=tk.NSEW, padx=5, pady=5)

    def markers_updated(self, marker_name, data_name, x, y, idx):
        """To be executed when the position of the markers is updated."""

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        current = self.datasets_var.get()
        self.update_velocities_list(current)

    def add_velocity(self):
        """Opens a dialog to add a new velocity to the list of velocities."""

    def remove_velocity(self):
        """Opens a dialog to remove the selected velocity to the list of velocities."""

    def switch_velocity(self):
        """Switch the plot to show the chosen velocity."""
        vel = self.velocities_var.get()
        dataset = self.datasets_var.get()

        if "global" in vel:
            self.populate_global_figure(dataset, vel)

    def populate_global_figure(self, dataset, vel_label):
        """Populates with data a global figure."""
        self.gfig = GlobalVelocity(self.visualise_frame, self.markers_updated)

        vels = self.data.velocities[dataset][vel_label]
        #        masks = None
        back = self.data.get_images(dataset, "MagZ")

        for i, label in enumerate((self.gfig.long, self.gfig.rad, self.gfig.circ)):
            self.gfig.update_line(label, (np.arange(len(vels[i])), vels[i]))
            self.gfig.update_bg(label, 0, back[0] / back[0].max())
            self.initial_marker_positions()

    def initial_marker_positions(self):
        """Finds the initial positions for the markers."""

    def update_velocities_list(self, dataset):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities.get(dataset, {})

        for v in self.velocities_frame.winfo_children()[2:]:
            v.grid_remove()

        for i, v in enumerate(velocities):
            ttk.Radiobutton(
                self.velocities_frame,
                text=v,
                value=v,
                variable=self.velocities_var,
                command=self.switch_velocity,
            ).grid(row=i, column=0, columnspan=2, sticky=tk.NSEW)

        if self.velocities_var.get() not in velocities and len(velocities) > 0:
            self.velocities_var.set(list(velocities.keys())[0])

    @trigger_event(name="calculate_velocities")
    def initialise_velocities(self, dataset):
        """Calculate pre-defined velocities if there are none for the chosen dataset."""
        if len(self.data.velocities.get(dataset, {})) > 0:
            return {}

        return dict(
            data=self.data,
            dataset_name=dataset,
            global_velocity=True,
            angular_regions=[6, 24],
            phantom=self.phantom_var.get(),
        )

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        # Include only datasets with a segmentation
        values = list(self.data.segments.keys())
        current = self.datasets_var.get()
        self.datasets_box.config(values=values)
        if current not in values:
            current = values[0]
        self.datasets_var.set(current)

        if len(self.data.velocities.get(current, {})) > 0:
            self.update_velocities_list(current)
            self.switch_velocity()
        else:
            self.initialise_velocities(current)

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass


class GlobalVelocity(object):

    long = "_long"
    rad = "_rad"
    circ = "_circ"

    def __init__(
        self,
        master=None,
        on_marker_moved: Optional[Callable] = None,
        figure_options: Optional[dict] = None,
        grid_options: Optional[dict] = None,
    ):

        figure_options = figure_options if figure_options is not None else {}
        grid_options = grid_options if grid_options is not None else {}

        fig_options = dict(constrained_layout=True)
        fig_options.update(figure_options)

        if master is not None:
            self.fig = Figure(**fig_options)
            _grid_options = dict(row=0, column=0, sticky=tk.NSEW)
            _grid_options.update(grid_options)
            canvas = FigureCanvasTkAgg(self.fig, master=master)
            canvas.get_tk_widget().grid(**_grid_options)
            master.rowconfigure(_grid_options["row"], weight=1)
            master.columnconfigure(_grid_options["column"], weight=1)
        else:
            self.fig = plt.figure(**fig_options)

        gs = self.fig.add_gridspec(2, 9, height_ratios=[4, 1])

        self.actions_manager = FigureActionsManager(self.fig, Markers)
        self.actions_manager.Markers.set_marker_moved(
            partial(self.on_marker_moved, on_marker_moved)
        )

        self.axes = self.add_velocity_subplots(gs)
        self.maps = self.add_maps_subplots(gs)
        self.vel_lines = self.add_velocity_lines()
        self.bg, self.vel_masks, self.cbar = self.add_bg_and_velocity_masks()
        self.add_markers()

        self.draw()

    def draw(self):
        """Convenience method for re-drawing the figure."""
        self.fig.canvas.draw()

    def add_velocity_subplots(self, gs):
        """Adds the velocity subplots, including ploting placeholders."""
        ax_long = self.fig.add_subplot(gs[0, :3])
        ax_rad = self.fig.add_subplot(gs[0, 3:6])
        ax_circ = self.fig.add_subplot(gs[0, 6:])

        ax_long.set_title("Longitudinal")
        ax_long.set_ylabel("Velocity (cm/s)")
        ax_long.set_xlabel("Time (ms)")
        ax_rad.set_title("Radial")
        ax_rad.set_xlabel("Time (ms)")
        ax_circ.set_title("Circumferential")
        ax_circ.set_xlabel("Time (ms)")

        return {"_long": ax_long, "_rad": ax_rad, "_circ": ax_circ}

    def add_velocity_lines(self):
        """Add lines to the velocity plots as placeholders"""
        vel_lines = {}
        vel_lines["_long"] = self.axes["_long"].plot([0], [0], "k", label="_long")[0]
        vel_lines["_rad"] = self.axes["_rad"].plot([0], [0], "k", label="_rad")[0]
        vel_lines["_circ"] = self.axes["_circ"].plot([0], [0], "k", label="_circ")[0]

        return vel_lines

    def add_maps_subplots(self, gs):
        """Adds the maps subplots."""
        maps = []
        for i in range(0, 3):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)

        for i in range(3, 6):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)

        for i in range(6, 9):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)

        for i, color in enumerate(["red", "green", "blue"] * 2):
            for side in ["left", "right", "bottom", "top"]:
                maps[i].spines[side].set_color(color)
                maps[i].spines[side].set_linewidth(2)

        for i, color in enumerate(["orange", "darkblue", "purple"]):
            for side in ["left", "right", "bottom", "top"]:
                maps[i + 6].spines[side].set_color(color)
                maps[i + 6].spines[side].set_linewidth(2)

        return maps

    def add_bg_and_velocity_masks(self):
        """Add bg and masks to the map subplots as placeholders"""
        bg = {"_long": [], "_rad": [], "_circ": []}
        masks = {"_long": [], "_rad": [], "_circ": []}
        cbar = {"_long": None, "_rad": None, "_circ": None}

        img = np.random.random((512, 512))

        for i in range(0, 3):
            bg["_long"].append(self.maps[i].imshow(img, cmap=plt.get_cmap("binary_r")))
            masks["_long"].append(
                self.maps[i].imshow(img * np.nan, cmap=plt.get_cmap("seismic"))
            )

        for i in range(3, 6):
            bg["_rad"].append(self.maps[i].imshow(img, cmap=plt.get_cmap("binary_r")))
            masks["_rad"].append(
                self.maps[i].imshow(img * np.nan, cmap=plt.get_cmap("seismic"))
            )

        for i in range(6, 9):
            bg["_circ"].append(self.maps[i].imshow(img, cmap=plt.get_cmap("binary_r")))
            masks["_circ"].append(
                self.maps[i].imshow(img * np.nan, cmap=plt.get_cmap("seismic"))
            )

        lims = [np.min(img), np.max(img)]
        cbar["_long"] = self.fig.colorbar(
            masks["_long"][0], ticks=lims, ax=self.maps[:3], orientation="horizontal"
        )
        cbar["_rad"] = self.fig.colorbar(
            masks["_rad"][0], ticks=lims, ax=self.maps[3:6], orientation="horizontal"
        )
        cbar["_circ"] = self.fig.colorbar(
            masks["_circ"][0], ticks=lims, ax=self.maps[6:9], orientation="horizontal"
        )

        return bg, masks, cbar

    def add_markers(self):
        """Adds markers to the plots."""
        add_marker = self.actions_manager.Markers.add_marker

        add_marker(self.vel_lines["_long"], "PS", color="red", marker="1")
        add_marker(self.vel_lines["_long"], "PD", color="green", marker="2")
        add_marker(self.vel_lines["_long"], "PAS", color="blue", marker="3")

        add_marker(self.vel_lines["_rad"], "PS", color="red", marker="1")
        add_marker(self.vel_lines["_rad"], "PD", color="green", marker="2")
        add_marker(self.vel_lines["_rad"], "PAS", color="blue", marker="3")
        add_marker(self.vel_lines["_rad"], "ES", color="black", marker="+")

        add_marker(self.vel_lines["_circ"], "PC1", color="orange", marker="1")
        add_marker(self.vel_lines["_circ"], "PC2", color="darkblue", marker="2")
        add_marker(self.vel_lines["_circ"], "PC3", color="purple", marker="3")

        self.axes["_long"].legend(frameon=False, markerscale=0.5)
        self.axes["_rad"].legend(frameon=False, markerscale=0.5)
        self.axes["_circ"].legend(frameon=False, markerscale=0.5)

    def update_marker_position(self, marker_label, data_label, new_x):
        """Convenience method to update the marker position."""
        self.actions_manager.Markers.update_marker_position(
            marker_label, data_label, new_x
        )

    def update_line(self, vel_label, data):
        """Updates the data of the chosen line."""
        self.update_data(self.vel_lines[vel_label], vel_label, data)

    def update_bg(self, vel_label, idx, data):
        """Updates the data of the chosen bg."""
        self.update_data(self.bg[vel_label][idx], vel_label, data)

    def update_mask(self, vel_label, idx, data):
        """Updates the data of the chosen bg."""
        self.update_data(self.vel_masks[vel_label][idx], vel_label, data)

    def update_data(self, subplot, vel_label, data):
        """Common data updating method."""
        subplot.set_data(data)
        self.axes[vel_label].relim()
        self.axes[vel_label].autoscale_view()
        self.draw()

    def on_marker_moved(self, callback, marker_name, data_name, x, y, idx):
        """When a marker moves, mask data should be updated."""
        marker_labels = {"PS": 0, "PD": 1, "PAS": 2, "PC1": 0, "PC2": 1, "PC3": 2}
        data = callback(marker_name, data_name, x, y, idx)
        if marker_labels.get(marker_name) is not None:
            self.update_mask(data_name, marker_labels[marker_name], data)
            self.draw()

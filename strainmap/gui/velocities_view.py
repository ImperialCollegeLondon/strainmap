import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view, trigger_event
from .figure_actions_manager import FigureActionsManager
from .figure_actions import Markers, ScrollFrames


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED

    def __init__(self, root):

        super().__init__(root, button_text="Velocities", button_image="speed.gif")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.visualise_frame = None
        self.datasets_box = None
        self.datasets_var = tk.StringVar(value="")
        self.velocities_frame = None
        self.velocities_var = tk.StringVar(value="")
        self.phantom_var = tk.BooleanVar(value=True)
        self.plot = None
        self.regional_fig = None
        self.param_tables = []
        self.current_region = 0
        self.images = None

        # Figure-related variables
        self.fig = None
        self.axes = None
        self.maps = None
        self.vel_lines = None
        self.bg_images = None
        self.vel_masks = None
        self.cbar = None
        self.marker_moved_info = ()

        self.create_controls()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self)
        control.columnconfigure(49, weight=1)
        self.visualise_frame = ttk.Frame(master=self)
        self.visualise_frame.columnconfigure(0, weight=1)
        self.visualise_frame.rowconfigure(0, weight=1)
        info = ttk.Frame(master=self)
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        info.columnconfigure(2, weight=1)

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

        # Information frame
        marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]
        for i in range(3):
            self.param_tables.append(ttk.Treeview(info, height=3))
            self.param_tables[-1]["columns"] = marker_lbl[3 * i : 3 * i + 3]
            self.param_tables[-1].heading("#0", text="")
            self.param_tables[-1].column("#0", minwidth=0, width=100, stretch=tk.YES)

            for j in range(3):
                self.param_tables[-1].heading(
                    marker_lbl[3 * i + j], text=marker_lbl[3 * i + j]
                )
                self.param_tables[-1].column(
                    marker_lbl[3 * i + j],
                    minwidth=0,
                    width=80,
                    stretch=tk.YES,
                    anchor=tk.E,
                )

        # Grid all the widgets
        control.grid(sticky=tk.NSEW, padx=10, pady=10)
        self.visualise_frame.grid(sticky=tk.NSEW, padx=10, pady=10)
        info.grid(sticky=tk.NSEW, padx=10, pady=10)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.velocities_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        for i in range(3):
            self.param_tables[i].grid(row=0, column=i, sticky=tk.NSEW, padx=5)

        # Bind the scrolling to change the regions in the plots
        self.bind_all("<MouseWheel>", self.scroll)

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        current = self.datasets_var.get()
        self.images = self.data.get_images(current, "MagZ")
        self.update_velocities_list(current)

    def add_velocity(self):
        """Opens a dialog to add a new velocity to the list of velocities."""

    def remove_velocity(self):
        """Opens a dialog to remove the selected velocity to the list of velocities."""

    def switch_velocity(self, region=0):
        """Switch the plot to show the chosen velocity."""
        self.current_region = region
        self.markers_figure(
            self.velocities, self.velocity_maps, self.images, self.markers
        )
        self.populate_tables()

    def marker_moved(self):
        """Updates plot and table after a marker has been moved."""
        self.populate_tables()
        if self.marker_moved_info[1] == 3:
            self.update_maps(self.velocity_maps, self.images, self.markers)
        else:
            axes, idx = self.marker_moved_info
            component = {"_long": 0, "_rad": 1, "_circ": 2}[axes]
            frame = int(self.markers[component, idx, 0])
            self.update_mask(axes, idx, self.velocity_maps[component, frame])
            self.update_bg(axes, idx, self.images[frame])
            self.draw()

        self.marker_moved_info = ()

    def scroll(self, *args):
        """Changes the region being plotted when scrolling with the mouse."""
        current_region = (self.current_region + 1) % len(
            self.data.velocities[self.datasets_var.get()][self.velocities_var.get()]
        )
        if self.current_region != self.current_region:
            self.switch_velocity(region=current_region)

    @property
    def velocities(self) -> np.ndarray:
        """Velocities of the current region."""
        return self.data.velocities[self.datasets_var.get()][self.velocities_var.get()][
            self.current_region
        ]

    @property
    def markers(self) -> np.ndarray:
        """Markers of the current region."""
        return self.data.markers[self.datasets_var.get()][self.velocities_var.get()][
            self.current_region
        ]

    @property
    def masks(self) -> np.ndarray:
        """Masks for the current region"""
        return (
            self.data.masks[self.datasets_var.get()][self.velocities_var.get()]
            != self.current_region + 1
        )

    @property
    def velocity_maps(self):
        """Calculate velocity maps out of the masks and cylindrical velocities."""
        cyl_label = f"cylindrical -{self.velocities_var.get().split('-')[-1]}"
        cylindrical = self.data.masks[self.datasets_var.get()][cyl_label]
        bmask = np.broadcast_to(self.masks, cylindrical.shape)
        return np.ma.masked_where(bmask, cylindrical)

    def populate_tables(self):
        """Populates the information tables with the marker parameters."""
        for t in self.param_tables:
            old_list = t.get_children()
            if len(old_list) > 0:
                t.delete(*old_list)

        for i, t in enumerate(self.param_tables):
            for k, text in enumerate(("Frame", "Velocity (cm/s)", "Norm. Time (s)")):
                val = np.around(self.markers[i, :3, k], decimals=(0, 2, 2)[k]).tolist()
                t.insert("", tk.END, text=text, values=val)

    def update_velocities_list(self, dataset):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities.get(dataset, {})

        for v in self.velocities_frame.winfo_children()[2:]:
            v.grid_remove()

        for i, v in enumerate(velocities):
            col, row = divmod(i, 3)
            ttk.Radiobutton(
                self.velocities_frame,
                text=v,
                value=v,
                variable=self.velocities_var,
                command=self.switch_velocity,
            ).grid(row=row, column=col, sticky=tk.NSEW)

        if self.velocities_var.get() not in velocities and len(velocities) > 0:
            self.velocities_var.set(list(velocities.keys())[0])

    @trigger_event(name="calculate_velocities")
    def initialise_velocities(self, dataset):
        """Calculate pre-defined velocities if there are none for the chosen dataset."""
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
        self.images = self.data.get_images(current, "MagZ")

        if len(self.data.velocities.get(current, {})) > 0:
            self.update_velocities_list(current)
            if self.marker_moved_info:
                self.marker_moved()
            else:
                self.switch_velocity()
        else:
            self.initialise_velocities(current)

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

    def markers_figure(
        self,
        velocities: np.ndarray,
        vel_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
    ):
        self.fig = Figure(constrained_layout=True)
        canvas = FigureCanvasTkAgg(self.fig, master=self.visualise_frame)
        canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)

        self.fig.actions_manager = FigureActionsManager(self.fig, Markers, ScrollFrames)
        self.fig.actions_manager.Markers.set_marker_moved(  # type: ignore
            self.update_marker
        )

        gs = self.fig.add_gridspec(2, 9, height_ratios=[4, 1])
        self.axes = self.add_velocity_subplots(gs)
        self.maps = self.add_maps_subplots(gs)
        self.vel_lines = self.add_velocity_lines(velocities)
        self.bg_images, self.vel_masks, self.cbar = self.images_and_velocity_masks(
            images, vel_masks, markers
        )
        self.add_markers(markers)

        self.draw()

    def draw(self):
        """Convenience method for re-drawing the figure."""
        self.fig.canvas.draw()

    def add_velocity_subplots(self, gs):
        """Adds the velocity subplots."""
        ax_long = self.fig.add_subplot(gs[0, :3])
        ax_rad = self.fig.add_subplot(gs[0, 3:6])
        ax_circ = self.fig.add_subplot(gs[0, 6:])

        ax_long.set_title("Longitudinal")
        ax_long.set_ylabel("Velocity (cm/s)")
        ax_long.set_xlabel("Frame")
        ax_rad.set_title("Radial")
        ax_rad.set_xlabel("Frame")
        ax_circ.set_title("Circumferential")
        ax_circ.set_xlabel("Frame")

        return {"_long": ax_long, "_rad": ax_rad, "_circ": ax_circ}

    def add_velocity_lines(self, vels):
        """Add lines to the velocity plots.

        vels - 2D array with the velocities with shape [components (3), frames]
        """
        x = np.arange(vels.shape[-1])
        return {
            label: self.axes[label].plot(x, vels[i], "k", label=label)[0]
            for i, label in enumerate(("_long", "_rad", "_circ"))
        }

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

    def images_and_velocity_masks(self, mag, vel_masks, markers):
        """Add bg and masks to the map subplots."""
        bg = {"_long": [], "_rad": [], "_circ": []}
        masks = {"_long": [], "_rad": [], "_circ": []}
        cbar = {"_long": None, "_rad": None, "_circ": None}

        xlim, ylim = self.find_limits(vel_masks[0, 0])
        for i in range(9):
            axes = ("_long", "_rad", "_circ")[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            bg[axes].append(
                self.maps[i].imshow(mag[frame], cmap=plt.get_cmap("binary_r"))
            )
            masks[axes].append(
                self.maps[i].imshow(
                    vel_masks[i // 3, frame], cmap=plt.get_cmap("seismic")
                )
            )
            self.maps[i].set_xlim(*xlim)
            self.maps[i].set_ylim(*ylim)

        cbar["_long"] = self.fig.colorbar(
            masks["_long"][0], ax=self.maps[:3], orientation="horizontal"
        )
        cbar["_rad"] = self.fig.colorbar(
            masks["_rad"][0], ax=self.maps[3:6], orientation="horizontal"
        )
        cbar["_circ"] = self.fig.colorbar(
            masks["_circ"][0], ax=self.maps[6:9], orientation="horizontal"
        )

        return bg, masks, cbar

    @staticmethod
    def find_limits(mask, margin=30):
        """Find the appropiate limits of a masked array in order to plot it nicely."""
        yaxes, xaxes = mask.nonzero()

        xmin = max(xaxes.min() - margin, 0)
        ymin = max(yaxes.min() - margin, 0)
        xmax = min(xaxes.max() + margin, mask.shape[0])
        ymax = min(yaxes.max() + margin, mask.shape[1])

        return (xmin, xmax), (ymax, ymin)

    def add_markers(self, markers):
        """Adds markers to the plots, assuming region 1 (the only one for global vel.).

        - markers - Contains all the marker information for all regions and components.
            Their shape is [regions, component (3), marker_id (3 or 4), marker_data (3)]
        """
        add_marker = self.fig.actions_manager.Markers.add_marker

        vel_lbl = ["_long"] * 3 + ["_rad"] * 3 + ["_circ"] * 3
        colors = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]

        for i in range(9):
            add_marker(
                self.vel_lines[vel_lbl[i]],
                xy=markers[i // 3, i % 3, :2],
                label=marker_lbl[i],
                color=colors[i],
                marker=str(i % 3 + 1),
            )

        add_marker(
            self.vel_lines["_rad"],
            xy=markers[1, 3, :2],
            label="ES",
            color="black",
            marker="+",
        )

        self.axes["_long"].legend(frameon=False, markerscale=0.5)
        self.axes["_rad"].legend(frameon=False, markerscale=0.5)
        self.axes["_circ"].legend(frameon=False, markerscale=0.5)

    def update_marker_position(self, marker_label, data_label, new_x):
        """Convenience method to update the marker position."""
        self.actions_manager.Markers.update_marker_position(
            marker_label, data_label, new_x
        )

    def update_line(self, vel_label, data, draw=False):
        """Updates the data of the chosen line."""
        self.update_data(self.vel_lines[vel_label], vel_label, data, draw)

    def update_bg(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.bg_images[vel_label][idx], vel_label, data, draw)

    def update_mask(self, vel_label, idx, data, draw=False):
        """Updates the data of the chosen bg."""
        self.update_data(self.vel_masks[vel_label][idx], vel_label, data, draw)

    def update_data(self, subplot, vel_label, data, draw=False):
        """Common data updating method."""
        subplot.set_data(data)
        self.axes[vel_label].relim()
        self.axes[vel_label].autoscale_view()

        if draw:
            self.draw()

    def update_maps(
        self, vel_masks: np.ndarray, images: np.ndarray, markers: np.ndarray
    ):
        """Updates the maps (masks and background data)."""
        for i in range(9):
            axes = ("_long", "_rad", "_circ")[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            self.update_mask(axes, i % 3, vel_masks[i // 3, frame])
            self.update_bg(axes, i % 3, images[frame])

        self.draw()

    @trigger_event
    def update_marker(self, marker, data, x, y, position):
        """When a marker moves, mask data should be updated."""
        marker_idx = {
            "PS": 0,
            "PD": 1,
            "PAS": 2,
            "PC1": 0,
            "PC2": 1,
            "PC3": 2,
            "ES": 3,
        }.get(marker.get_label())
        component = {"_long": 0, "_rad": 1, "_circ": 2}.get(data.get_label())
        self.marker_moved_info = (data.get_label(), marker_idx)
        return dict(
            data=self.data,
            dataset=self.datasets_var.get(),
            vel_label=self.velocities_var.get(),
            region=self.current_region,
            component=component,
            marker_idx=marker_idx,
            position=position,
        )

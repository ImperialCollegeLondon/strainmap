import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view, trigger_event
from .figure_actions_manager import FigureActionsManager
from .figure_actions import Markers, SimpleScroller


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
        self.limits = None
        self.marker_artists = None
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
        marker_lbl = (("PS", "PD", "PAS"), ("PS", "PD", "PAS"), ("PC1", "PC2", "PC3"))
        for i in range(3):
            self.param_tables.append(ttk.Treeview(info, height=14))
            self.param_tables[-1].tag_configure("current", background="#f8d568")
            self.param_tables[-1].tag_configure("others", background="#FFFFFF")
            self.param_tables[-1]["columns"] = marker_lbl[i]
            self.param_tables[-1].heading("#0", text="Region")
            self.param_tables[-1].column("#0", width=100, stretch=tk.YES)

            for j in range(3):
                self.param_tables[-1].heading(marker_lbl[i][j], text=marker_lbl[i][j])
                self.param_tables[-1].column(
                    marker_lbl[i][j], width=80, stretch=tk.YES, anchor=tk.E
                )

        # Grid all the widgets
        control.grid(sticky=tk.NSEW, padx=10, pady=5)
        self.visualise_frame.grid(sticky=tk.NSEW, padx=10, pady=5)
        info.grid(sticky=tk.NSEW, padx=10, pady=10)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.velocities_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        for i in range(3):
            self.param_tables[i].grid(row=0, column=i, sticky=tk.NSEW, padx=5)

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        current = self.datasets_var.get()
        if self.data.velocities[current]:
            self.images = self.data.get_images(current, "MagZ")
            self.update_velocities_list(current)
            if self.marker_moved_info:
                self.marker_moved()
            else:
                self.switch_velocity()
        else:
            self.initialise_velocities(current)

    def add_velocity(self):
        """Opens a dialog to add a new velocity to the list of velocities."""

    def remove_velocity(self):
        """Opens a dialog to remove the selected velocity to the list of velocities."""

    def switch_velocity(self):
        """Switch the plot to show the chosen velocity."""
        if self.fig is None:
            self.current_region = 0
            self.markers_figure(
                self.velocities,
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
            )
        else:
            self.current_region = -1
            self.scroll()
            self.draw()

        self.populate_tables()

    def marker_moved(self):
        """Updates plot and table after a marker has been moved."""
        self.populate_tables()
        if self.marker_moved_info[1] == 3:
            self.update_maps(
                self.velocity_maps, self.images, self.markers[self.current_region]
            )
        else:
            self.update_one_map(
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
                *self.marker_moved_info,
            )

        self.marker_moved_info = ()

    def scroll(self, step=1, *args):
        """Changes the region being plotted when scrolling with the mouse."""
        current_region = (self.current_region + step) % self.regions

        if self.current_region != current_region:
            self.fig.actions_manager.SimpleScroller.disabled = True
            self.current_region = current_region
            self.update_velocities(self.velocities, draw=False)
            self.update_maps(
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
                draw=False,
            )
            self.update_markers(self.markers[self.current_region], draw=False)
            self.populate_tables()
            self.fig.actions_manager.SimpleScroller.disabled = False

        return self.current_region, None, None

    @property
    def regions(self) -> int:
        """Number of regions for the selected velocity."""
        return len(
            self.data.velocities[self.datasets_var.get()][self.velocities_var.get()]
        )

    @property
    def velocities(self) -> np.ndarray:
        """Velocities of the current region."""
        return self.data.velocities[self.datasets_var.get()][self.velocities_var.get()][
            self.current_region
        ]

    @property
    def markers(self) -> np.ndarray:
        """Markers of the current region."""
        return self.data.markers[self.datasets_var.get()][self.velocities_var.get()]

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

    @staticmethod
    def region_labels(regions):
        """Provides the region labels, if any."""
        if regions == 6:
            return "AS", "A", "AL", "IL", "I", "IS"
        else:
            return list(range(1, regions + 1))

    def populate_tables(self):
        """Populates the information tables with the marker parameters."""
        for t in self.param_tables:
            old_list = t.get_children()
            if len(old_list) > 0:
                t.delete(*old_list)

        labels = self.region_labels(len(self.markers))
        for i, t in enumerate(self.param_tables):
            vel = t.insert("", tk.END, text="Velocity (cm/s)", open=True)
            time = t.insert("", tk.END, text="Norm. Time (s)", open=True)
            for j, marker in enumerate(self.markers):
                tag = "current" if j == self.current_region else "others"
                val = np.around(marker[i, :3, 1], decimals=2).tolist()
                t.insert(vel, tk.END, text=labels[j], values=val, tags=(tag,))
                val = np.around(marker[i, :3, 2], decimals=2).tolist()
                t.insert(time, tk.END, text=labels[j], values=val, tags=(tag,))

    def update_velocities_list(self, dataset):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities[dataset]

        for v in self.velocities_frame.winfo_children():
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

        if self.data.velocities[current]:
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

        self.fig.actions_manager = FigureActionsManager(
            self.fig, Markers, SimpleScroller
        )
        self.fig.actions_manager.Markers.set_marker_moved(  # type: ignore
            self.update_marker
        )
        self.fig.actions_manager.SimpleScroller.set_scroller(  # type: ignore
            self.scroll
        )

        gs = self.fig.add_gridspec(2, 9, height_ratios=[6, 2])
        self.axes = self.add_velocity_subplots(gs)
        self.maps = self.add_maps_subplots(gs)
        self.vel_lines = self.add_velocity_lines(velocities)
        self.bg_images, self.vel_masks, self.cbar = self.images_and_velocity_masks(
            images, vel_masks, markers
        )
        self.marker_artists = self.add_markers(markers)

        self.draw()

    def draw(self):
        """Convenience method for re-drawing the figure."""
        self.fig.canvas.draw_idle()

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
        colours = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        for i, color in enumerate(colours):
            maps.append(self.fig.add_subplot(gs[1, i]))
            maps[-1].get_xaxis().set_visible(False)
            maps[-1].get_yaxis().set_visible(False)
            for side in ["left", "right", "bottom", "top"]:
                maps[-1].spines[side].set_color(color)
                maps[-1].spines[side].set_linewidth(3)

        return maps

    def images_and_velocity_masks(self, mag, vel_masks, markers):
        """Add bg and masks to the map subplots."""
        bg = {"_long": [], "_rad": [], "_circ": []}
        masks = {"_long": [], "_rad": [], "_circ": []}

        if "global" in self.velocities_var.get():
            self.limits = self.find_limits(vel_masks[0, 0])

        vmin, vmax = vel_masks.min(), vel_masks.max()
        for i in range(9):
            axes = ("_long", "_rad", "_circ")[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            bg[axes].append(
                self.maps[i].imshow(mag[frame], cmap=plt.get_cmap("binary_r"))
            )
            masks[axes].append(
                self.maps[i].imshow(
                    vel_masks[i // 3, frame],
                    cmap=plt.get_cmap("seismic"),
                    vmin=vmin,
                    vmax=vmax,
                )
            )
            self.maps[i].set_xlim(*self.limits[0])
            self.maps[i].set_ylim(*self.limits[1])

        cbar = self.fig.colorbar(masks["_long"][0], ax=self.maps[0], pad=-1.4)

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
        """Adds markers to the plots.).

        - markers - Contains all the marker information for all regions and components.
            Their shape is [regions, component (3), marker_id (3 or 4), marker_data (3)]
        """
        add_marker = self.fig.actions_manager.Markers.add_marker

        vel_lbl = ["_long"] * 3 + ["_rad"] * 3 + ["_circ"] * 3
        colors = ["red", "green", "blue"] * 2 + ["orange", "darkblue", "purple"]
        marker_lbl = ["PS", "PD", "PAS"] * 2 + ["PC1", "PC2", "PC3"]

        markers_artists = []
        for i in range(9):
            markers_artists.append(
                add_marker(
                    self.vel_lines[vel_lbl[i]],
                    xy=markers[i // 3, i % 3, :2],
                    label=marker_lbl[i],
                    color=colors[i],
                    marker=str(i % 3 + 1),
                    markeredgewidth=2,
                )
            )

        markers_artists.append(
            add_marker(
                self.vel_lines["_rad"],
                xy=markers[1, 3, :2],
                label="ES",
                color="black",
                marker="+",
                markeredgewidth=2,
            )
        )

        self.axes["_long"].legend(frameon=False, markerscale=0.5)
        self.axes["_rad"].legend(frameon=False, markerscale=0.5)
        self.axes["_circ"].legend(frameon=False, markerscale=0.5)

        return markers_artists

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
        self.axes[vel_label].autoscale()

        if draw:
            self.draw()

    def update_maps(
        self, vel_masks: np.ndarray, images: np.ndarray, markers: np.ndarray, draw=True
    ):
        """Updates the maps (masks and background data)."""
        for i in range(9):
            axes = ("_long", "_rad", "_circ")[i // 3]
            frame = int(markers[i // 3, i % 3, 0])
            self.update_mask(axes, i % 3, vel_masks[i // 3, frame])
            self.update_bg(axes, i % 3, images[frame])

        if draw:
            self.draw()

    def update_one_map(
        self,
        vel_masks: np.ndarray,
        images: np.ndarray,
        markers: np.ndarray,
        axes: str,
        idx: int,
    ):
        """Updates the maps correspoinding to a single marker."""
        component = {"_long": 0, "_rad": 1, "_circ": 2}[axes]
        frame = int(markers[component, idx, 0])
        self.update_mask(axes, idx, vel_masks[component, frame])
        self.update_bg(axes, idx, images[frame])
        self.draw()

    def update_velocities(self, vels, draw=True):
        """Updates all velocities."""
        x = np.arange(vels.shape[-1])

        for i, label in enumerate(("_long", "_rad", "_circ")):
            self.update_line(label, (x, vels[i]))

        if draw:
            self.draw()

    def update_markers(self, markers, draw=True):
        """Updates the position of all markers in a figure."""
        update_position = self.fig.actions_manager.Markers.update_marker_position
        for i in range(9):
            update_position(self.marker_artists[i], int(markers[i // 3, i % 3, 0]))

        update_position(self.marker_artists[-1], int(markers[1, 3, 0]))

        if draw:
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

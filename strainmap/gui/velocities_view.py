import re
import tkinter as tk
import tkinter.filedialog
from tkinter import ttk

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .base_window_and_task import Requisites, TaskViewBase, register_view
from ..coordinates import Region, Comp


def get_sa_location(cine):
    pattern = r"[sS][aA]([0-9])"
    m = re.search(pattern, cine)
    return int(m.group(1)) if hasattr(m, "group") else 99


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED
    axes_lbl = ("_long", "_rad", "_circ")
    marker_idx = {"PS": 0, "PD": 1, "PAS": 2, "PC1": 0, "PC2": 1, "PC3": 2, "ES": 3}

    def __init__(self, root, controller):

        super().__init__(
            root,
            controller,
            button_text="Velocities",
            button_image="speed.gif",
            button_row=2,
        )
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.visualise_frame = None
        self.cines_box = None
        self.cines_var = tk.StringVar(value="")
        self.velocities_frame = None
        self.velocities_var = tk.StringVar(value="global")
        self.plot = None
        self.regional_fig = None
        self.param_tables = []
        self.current_region = 0
        self.update_vel_btn = None
        self.export_btn = None
        self.export_super_btn = None
        self.reverse_vel_var = (
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
            tk.BooleanVar(value=False),
        )
        self.reverse_status = (False, False, False)
        self.orientation_var = tk.StringVar(value=self.data.orientation)

        # Figure-related variables
        self.fig = None
        self.axes = None
        self.maps = None
        self.vel_lines = None
        self.vel_masks = None
        self.cbar = None
        self.limits = None
        self.marker_artists = None
        self.vel_lim = dict()
        self.fixed_markers = []

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
        cine_frame = ttk.Labelframe(control, text="Datasets:", borderwidth=0)
        cine_frame.columnconfigure(0, weight=1)

        self.cines_box = ttk.Combobox(
            master=cine_frame, textvariable=self.cines_var, values=[], state="readonly"
        )
        self.cines_box.bind("<<ComboboxSelected>>", self.cine_changed)

        # Velocities frame
        self.velocities_frame = ttk.Labelframe(control, text="Velocities:")

        # Information frame
        marker_lbl = (("PS", "PD", "PAS"), ("PS", "PD", "PAS"), ("PC1", "PC2", "PC3"))
        for labels in marker_lbl:
            self.param_tables.append(ttk.Treeview(info, height=8))
            self.param_tables[-1].tag_configure("current", background="#f8d568")
            self.param_tables[-1].tag_configure("others", background="#FFFFFF")
            self.param_tables[-1]["columns"] = labels
            self.param_tables[-1].heading("#0", text="Region")
            self.param_tables[-1].column("#0", width=110, stretch=tk.YES)

            for l in labels:
                self.param_tables[-1].heading(l, text=l)
                self.param_tables[-1].column(l, width=80, stretch=tk.YES, anchor=tk.E)

        # Sign reversal frame
        reversal_frame = ttk.Labelframe(control, text="Reverse sign:")
        x = ttk.Checkbutton(
            reversal_frame,
            text="X",
            variable=self.reverse_vel_var[0],
            command=self.reversal_checked,
        )
        y = ttk.Checkbutton(
            reversal_frame,
            text="Y",
            variable=self.reverse_vel_var[1],
            command=self.reversal_checked,
        )
        z = ttk.Checkbutton(
            reversal_frame,
            text="Z",
            variable=self.reverse_vel_var[2],
            command=self.reversal_checked,
        )
        self.update_vel_btn = ttk.Button(
            control,
            text="Update",
            command=self.recalculate_velocities,
            state="disabled",
        )

        orientation_frame = ttk.Labelframe(control, text="Orientation:")
        ttk.Radiobutton(
            orientation_frame,
            text="CW",
            value="CW",
            variable=self.orientation_var,
            command=self.change_orientation,
        ).grid(row=0, column=0, sticky=tk.NSEW)
        ttk.Radiobutton(
            orientation_frame,
            text="CCW",
            value="CCW",
            variable=self.orientation_var,
            command=self.change_orientation,
        ).grid(row=0, column=1, sticky=tk.NSEW)

        self.export_btn = ttk.Button(
            control, text="To Excel", command=self.export, state="disabled"
        )
        self.export_super_btn = ttk.Button(
            control,
            text="Export superpixels",
            command=self.export_superpixel,
            state="disabled",
        )

        # Grid all the widgets
        control.grid(row=0, column=0, sticky=tk.NSEW, pady=5)
        self.visualise_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        info.grid(row=2, column=0, sticky=tk.NSEW, pady=5)
        cine_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        self.cines_box.grid(row=0, column=0, sticky=tk.NSEW)
        reversal_frame.grid(row=0, column=98, rowspan=2, sticky=tk.NSEW, padx=5)
        x.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        y.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        z.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        self.update_vel_btn.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        orientation_frame.grid(row=0, column=3, sticky=tk.NSEW, padx=5)
        self.velocities_frame.grid(row=0, column=4, sticky=tk.NSEW, padx=5)
        self.export_btn.grid(row=0, column=98, sticky=tk.NSEW, padx=5)
        self.export_super_btn.grid(row=0, column=99, sticky=tk.NSEW, padx=5)

        for i, table in enumerate(self.param_tables):
            table.grid(row=0, column=i, sticky=tk.NSEW, padx=5)

    def cine_changed(self, *args):
        """Updates the view when the selected cine is changed."""
        current = self.cines_var.get()
        if current in self.data.velocities.cine:
            self.update_velocities_list(current)
        else:
            self.calculate_velocities(current)

        self.replot()

    def recalculate_velocities(self):
        """Recalculate velocities after a sign reversal."""
        self.update_vel_btn.state(["disabled"])
        cine = self.cines_var.get()
        self.calculate_velocities(cine)
        self.replot()

    def reversal_checked(self):
        """Enables/disables de update velocities button if amy sign reversal changes."""
        if tuple(var.get() for var in self.reverse_vel_var) != self.data.sign_reversal:
            self.update_vel_btn.state(["!disabled"])
        else:
            self.update_vel_btn.state(["disabled"])

    def change_orientation(self):
        """Change the orientation CW <-> CCW of the angular regions of all cines"""
        if self.orientation_var.get() != self.data.orientation:
            self.data.set_orientation(self.orientation_var.get())
            if "24" in self.velocities_var.get():
                self.replot()
            else:
                self.populate_tables()

    def find_velocity_limits(self, vel_label: Region):
        """Finds suitable maximum and minimum for the velocity plots."""
        vel = self.data.velocities.sel(cine=self.cines_var.get(), region=vel_label)
        for i, label in enumerate((Comp.LONG, Comp.RAD, Comp.CIRC)):
            mini = vel.sel(comp=label).min().item()
            maxi = vel.sel(comp=label).max().item()
            m = (maxi - mini) * 0.10
            self.vel_lim[label] = (mini - m, maxi + m)

    def display_plots(self, show=True):
        """Show/hide the plots"""
        if show:
            self.visualise_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
            self.export_btn.state(["!disabled"])
            self.export_super_btn.state(["!disabled"])
            self.controller.window.update()
        else:
            self.visualise_frame.grid_forget()
            self.export_btn.state(["disabled"])
            self.export_super_btn.state(["disabled"])
            self.controller.window.update()

    def replot(self):
        """Updates the plot to show the chosen velocity."""
        cine = self.cines_var.get()
        vel_label = self.velocities_var.get().upper().replace(" ", "_")

        self.find_velocity_limits(vel_label)
        if vel_label == Region.ANGULAR_x24:
            self.color_plots(cine)
        elif self.fig is None or self.current_region == -1:
            self.current_region = 0
            self.markers_figure(
                self.velocities,
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
            )
            self.populate_tables()
        else:
            self.current_region = -1
            self.scroll()
            self.draw()
            self.populate_tables()

        self.display_plots(True)

    def color_plots(self, cine):
        """Creates the color plots for the case of 24 angular regions."""
        gmark = self.data.markers.sel(cine=cine, region=Region.GLOBAL, quantity="frame")
        markers_idx = gmark[:, :3, 0].flatten()
        self.fig = colour_figure(
            self.data.velocities.sel(cine=cine, region=Region.ANGULAR_x24),
            self.region_labels(6),
            markers_idx,
            self.visualise_frame,
        )
        self.fig.canvas.draw_idle()
        self.current_region = -1
        markers = self.data.markers.sel(cine=cine, region=Region.ANGULAR_x6)
        self.populate_tables(markers)

    def marker_moved(self, table, marker):
        """Updates plot and table after a marker has been moved."""
        if marker == "ES":
            self.update_table_es()
            for f in self.fixed_markers:
                f.set_xdata([self.es_marker[0]] * 2)
        else:
            self.update_table_one_marker(table, marker)
            self.update_one_map(
                self.velocity_maps,
                self.images,
                self.markers[self.current_region],
                table,
                marker,
            )

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
            for vel_label, ax in self.axes.items():
                ax.set_ylim(*self.vel_lim[vel_label])

        return self.current_region, None, None

    @property
    def images(self) -> xr.DataArray:
        current = self.cines_var.get()
        return self.data.data_files.mag(current)

    @property
    def regions(self) -> int:
        """Number of regions for the selected velocity."""
        return len(
            self.data.velocities[self.cines_var.get()][self.velocities_var.get()]
        )

    @property
    def velocities(self) -> np.ndarray:
        """Velocities of the current region."""
        return self.data.velocities[self.cines_var.get()][self.velocities_var.get()][
            self.current_region
        ]

    @property
    def markers(self) -> np.ndarray:
        """Markers of the current region."""
        return self.data.markers[self.cines_var.get()][self.velocities_var.get()]

    @property
    def masks(self) -> np.ndarray:
        """Masks for the current region"""
        return (
            self.data.masks[self.cines_var.get()][self.velocities_var.get()]
            != self.current_region + 1
        )

    @property
    def velocity_maps(self):
        """Calculate velocity maps out of the masks and cylindrical velocities."""
        cylindrical = self.data.masks[self.cines_var.get()]["cylindrical"]
        bmask = np.broadcast_to(self.masks, cylindrical.shape)
        return np.ma.masked_where(bmask, cylindrical)

    def region_labels(self, regions):
        """Provides the region labels, if any."""
        if regions == 6:
            labels = "AS", "A", "AL", "IL", "I", "IS"
            if self.data.orientation == "CCW":
                labels = labels[::-1]
            return labels
        else:
            return list(range(1, regions + 1))

    def populate_tables(self, markers=None):
        """Populates the information tables with the marker parameters."""
        markers = markers if markers is not None else self.markers
        for t in self.param_tables:
            old_list = t.get_children()
            if len(old_list) > 0:
                t.delete(*old_list)

        labels = self.region_labels(len(markers))
        for i, t in enumerate(self.param_tables):
            vel = t.insert("", tk.END, text="Velocity (cm/s)", open=True)
            time = t.insert("", tk.END, text="Norm. Time (ms)", open=True)
            for j, marker in enumerate(markers):
                tag = "current" if j == self.current_region else "others"
                val = np.around(marker[i, :3, 1], decimals=2).tolist()
                t.insert(vel, tk.END, text=labels[j], values=val, tags=(tag,))
                val = np.around(marker[i, :3, 2], decimals=2).tolist()
                t.insert(time, tk.END, text=labels[j], values=val, tags=(tag,))

    def update_table_one_marker(self, table, marker):
        """Updates peak velocity and time table entry for a single marker."""
        table = self.axes_lbl.index(table)
        idx = self.marker_idx[marker]
        t = self.param_tables[table]
        velitem = t.get_children(t.get_children()[0])[self.current_region]
        timeitem = t.get_children(t.get_children()[1])[self.current_region]

        t.set(
            velitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 1], 2),
        )
        t.set(
            timeitem,
            column=marker,
            value=round(self.markers[self.current_region, table, idx, 2], 2),
        )

    def update_table_es(self):
        """Updates a row of the markers table after ES has moved."""
        for i, t in enumerate(self.param_tables):
            for r in range(len(self.markers)):
                timeitem = t.get_children(t.get_children()[1])[r]

                t.item(
                    timeitem,
                    values=np.around(self.markers[r, i, :3, 2], decimals=2).tolist(),
                )

    def update_velocities_list(self, cine):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities.sel(cine=cine)

        for v in self.velocities_frame.winfo_children():
            v.grid_remove()

        vel_list = np.unique(velocities.region).data
        vel_list = sorted(vel_list, reverse=True)
        for i, v in enumerate(vel_list):
            text = v.split(" - ")[0]
            ttk.Radiobutton(
                self.velocities_frame,
                text=text,
                value=v.name.lower().replace("_", " "),
                variable=self.velocities_var,
                command=self.replot,
            ).grid(row=0, column=i, sticky=tk.NSEW)

        self.velocities_var.set("global")

    def calculate_velocities(self, cine):
        """Calculate pre-defined velocities for the chosen cine."""
        self.display_plots(False)
        self.controller.calculate_velocities(
            cine_name=cine,
            sign_reversal=tuple(-1 if var.get() else 1 for var in self.reverse_vel_var),
        )
        self.update_velocities_list(self.cines_var.get())

    def update_marker(self, marker, data, x, y, position):
        """When a marker moves, mask data should be updated."""
        self.controller.update_marker(
            cine=self.cines_var.get(),
            vel_label=self.velocities_var.get(),
            region=self.current_region,
            component=self.axes_lbl.index(data.get_label()),
            marker_idx=self.marker_idx[marker.get_label()],
            position=position,
        )
        self.marker_moved(data.get_label(), marker.get_label())

    def export(self, *args):
        """Exports the current velocity data to an XLSX file."""
        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_{self.cines_var.get()}_velocity.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            self.controller.export_velocity(
                filename=filename,
                cine=self.cines_var.get(),
                vel_label=self.velocities_var.get(),
            )

    def export_superpixel(self, *args):
        """ Exports the current superpixel velocity data to an XLSX file.

        TODO: Remove in final version
        """
        from ..models.writers import export_superpixel

        meta = self.data.metadata()
        name, date = [meta[key] for key in ["Patient Name", "Date of Scan"]]
        init = f"{name}_{date}_{self.cines_var.get()}_velocity_super.xlsx"

        filename = tk.filedialog.asksaveasfilename(
            initialfile=init,
            defaultextension="xlsx",
            filetypes=[("Excel files", "*.xlsx")],
        )
        if filename != "":
            export_superpixel(
                data=self.data, cine=self.cines_var.get(), filename=filename
            )

    def populate_cine_box(self):
        """Populate the cine box with available segmentations."""
        values = sorted(self.data.segments.cine.data, key=get_sa_location)
        current = self.cines_var.get()
        self.cines_box.config(values=values)
        if current not in values:
            self.cines_var.set(values[0])

    def update_sign_reversal(self):
        """Updates the sign reversal information with data.sign_reversal info."""
        for i, var in enumerate(self.data.sign_reversal):
            self.reverse_vel_var[i].set(var == -1)

    def update_widgets(self):
        """ Updates widgets after an update in the data var. """
        self.populate_cine_box()
        self.update_sign_reversal()
        self.cine_changed()

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

    @property
    def es_marker(self):
        """ ES marker position. """
        return self.data.markers[self.cines_var.get()]["global"][0, 1, 3, :2]


def colour_figure(
    velocities: xr.DataArray,
    labels: tuple,
    markers_idx: xr.DataArray,
    master: ttk.Frame,
) -> Figure:
    """Creates the color plots for the regional velocities."""
    fig = Figure(constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
    ax = fig.subplots(ncols=3, nrows=1)

    n_reg = velocities.sizes["region"]
    space = n_reg / len(labels)
    lines_pos = np.arange(space, n_reg, space) - 0.5
    labels_pos = np.arange(space // 2, n_reg, space) - 0.5

    for i, comp in enumerate((Comp.LONG, Comp.RAD, Comp.CIRC)):
        ax[i].imshow(
            velocities.sel(comp=comp),
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="bilinear",
        )
        ax[i].set_title(comp.value)

        ax[i].set_yticks(lines_pos, minor=True)
        ax[i].set_yticks(labels_pos, minor=False)
        ax[i].set_yticklabels(labels[::-1], minor=False)
        ax[i].yaxis.grid(True, which="minor", color="k", linestyle="-")
        ax[i].set_ylim((-0.5, n_reg - 0.5))

        idx = markers_idx.sel(comp=comp).dropna(dim="marker")
        ax[i].set_xticks(idx, minor=False)
        ax[i].set_xticklabels([m.item().name for m in idx.marker], minor=False)

        fig.colorbar(ax[i].images[0], ax=ax[i], orientation="horizontal")

    return fig

import tkinter as tk
import tkinter.filedialog
from tkinter import ttk
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..coordinates import Comp, Mark, Region
from ..tools import get_sa_location
from .base_window_and_task import Requisites, TaskViewBase, register_view
from .markers_figure import MarkersFigure


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED
    ids: Dict[str, Sequence[str]] = {
        Comp.LONG.name: (Mark.PS.name, Mark.PD.name, Mark.PAS.name),
        Comp.RAD.name: (Mark.PS.name, Mark.PD.name, Mark.PAS.name),
        Comp.CIRC.name: (Mark.PC1.name, Mark.PC2.name, Mark.PC3.name),
    }

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
        self.param_tables: Dict[str, ttk.Treeview] = {}
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
        self.fig: Optional[MarkersFigure, Figure] = None

        self.create_controls()

    @property
    def vel_var(self) -> str:
        """Convenience method to get the selected velocity"""
        return self.velocities_var.get().upper().replace("_X", "_x")

    @property
    def images(self) -> xr.DataArray:
        cine = self.cines_var.get()
        return self.data.data_files.images(cine).sel(comp=Comp.MAG.name)

    @property
    def regions(self) -> int:
        """Number of regions for the selected velocity."""
        region = Region[self.vel_var]
        return region.value

    @property
    def markers(self) -> xr.DataArray:
        """Markers of the current region."""
        return self.data.markers.sel(cine=self.cines_var.get(), region=self.vel_var)

    @property
    def masks(self) -> np.ndarray:
        """Masks for the current region"""
        return (
            self.data.masks.sel(cine=self.cines_var.get(), region=self.vel_var)
            != self.current_region + 1
        )

    @property
    def cylindrical(self):
        """Cylindrical velocities for the current cine and region."""
        cine = self.cines_var.get()
        return self.data.cylindrical.sel(cine=cine).where(
            self.data.masks.sel(cine=cine, region=self.vel_var)
        )

    @property
    def velocities(self):
        """Cylindrical velocities for the current cine and region."""
        return self.data.velocities.sel(cine=self.cines_var.get(), region=self.vel_var)

    def create_controls(self):
        """Creates all the widgets of the view."""
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
        for comp, markers in self.ids.items():
            self.param_tables[comp] = ttk.Treeview(info, height=8)
            self.param_tables[comp].tag_configure("current", background="#f8d568")
            self.param_tables[comp].tag_configure("others", background="#FFFFFF")
            self.param_tables[comp]["columns"] = [m for m in markers]
            self.param_tables[comp].heading("#0", text="Region")
            self.param_tables[comp].column("#0", width=110, stretch=tk.YES)

            for m in markers:
                self.param_tables[comp].heading(m, text=m)
                self.param_tables[comp].column(m, width=80, stretch=tk.YES, anchor=tk.E)

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
        reversal_frame.grid(row=0, column=97, rowspan=2, sticky=tk.NSEW, padx=5)
        x.grid(row=0, column=0, sticky=tk.NSEW, padx=5)
        y.grid(row=0, column=1, sticky=tk.NSEW, padx=5)
        z.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        self.update_vel_btn.grid(row=0, column=2, sticky=tk.NSEW, padx=5)
        orientation_frame.grid(row=0, column=3, sticky=tk.NSEW, padx=5)
        self.velocities_frame.grid(row=0, column=4, sticky=tk.NSEW, padx=5)
        self.export_btn.grid(row=0, column=98, sticky=tk.NSEW, padx=5)
        self.export_super_btn.grid(row=0, column=99, sticky=tk.NSEW, padx=5)

        for i, table in enumerate(self.param_tables.values()):
            table.grid(row=0, column=i, sticky=tk.NSEW, padx=5)

    def cine_changed(self, *args):
        """Updates the view when the selected cine is changed."""
        self.controller.progress("Changing selected cine...")
        current = self.cines_var.get()
        if (
            "cine" not in self.data.velocities.dims
            or current not in self.data.velocities.cine
        ):
            self.calculate_velocities(current)
        else:
            self.update_velocities_list(current)

        for i, var in enumerate(self.reverse_vel_var):
            var.set(True if self.data.sign_reversal.data[i] == -1 else False)

        self.replot()
        self.controller.progress("Done!")

    def recalculate_velocities(self):
        """Recalculate velocities after a sign reversal."""
        self.update_vel_btn.state(["disabled"])
        for cine in self.data.cylindrical.cine:
            self.calculate_velocities(cine.item(), update=True)
        self.fig.line_limits = self.find_velocity_limits()
        self.replot()

    def reversal_checked(self):
        """Enables/disables de update velocities button if amy sign reversal changes."""
        if tuple(-1 if var.get() else 1 for var in self.reverse_vel_var) != tuple(
            self.data.sign_reversal
        ):
            self.update_vel_btn.state(["!disabled"])
        else:
            self.update_vel_btn.state(["disabled"])

    def change_orientation(self):
        """Change the orientation CW <-> CCW of the angular regions of all cines"""
        if self.orientation_var.get() != self.data.orientation:
            self.data.set_orientation(self.orientation_var.get())
            if "24" in self.vel_var:
                self.replot()
            else:
                self.populate_tables()

    def find_velocity_limits(self):
        """Finds suitable maximum and minimum for the velocity plots."""
        vel = self.data.velocities.sel(cine=self.cines_var.get())
        result: Dict[str, Tuple[float, float]] = {}
        for i, label in enumerate((Comp.LONG.name, Comp.RAD.name, Comp.CIRC.name)):
            mini = vel.sel(comp=label).min().item()
            maxi = vel.sel(comp=label).max().item()
            m = (maxi - mini) * 0.10
            result[label] = (mini - m, maxi + m)
        return result

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
        region = Region[self.vel_var]

        if region == Region.ANGULAR_x24:
            self.create_color_plot()

        elif self.fig is None or self.current_region == -1:
            self.create_markers_plot()

        else:
            self.update_plot()

        self.display_plots(True)

    def create_color_plot(self) -> None:
        """Creates the color plots for the case of 24 angular regions.

        After creating the plot, tables are updated.

        Returns:
            None
        """
        cine = self.cines_var.get()
        self.current_region = -1
        self.fig = colour_figure(
            self.data.velocities.sel(cine=cine, region=Region.ANGULAR_x24.name),
            self.region_labels(6),
            self.data.markers.sel(
                cine=cine, region=Region.GLOBAL.name, quantity="frame"
            ),
            self.visualise_frame,
        )
        markers = self.data.markers.sel(cine=cine, region=Region.ANGULAR_x6.name)
        self.populate_tables(markers)

    def create_markers_plot(self) -> None:
        """Plot the velocity curves, color maps and markers for the chosen cine/region.

        After creating the plot, tables are updated.

        Returns:
            None
        """
        self.current_region = 0
        self.fig = MarkersFigure(
            self.velocities.isel(region=self.current_region, missing_dims="ignore"),
            self.cylindrical.isel(region=self.current_region, missing_dims="ignore"),
            self.images,
            self.markers.isel(region=self.current_region, missing_dims="ignore"),
            master=self.visualise_frame,
            line_limits=self.find_velocity_limits(),
            ids=self.ids,
        )
        self.fig.set_scroller(self.scroll)
        self.fig.set_marker_moved(self.update_marker)
        self.populate_tables()

    def update_plot(self):
        """Update the existing plot and tables after scrolling.

        Returns:
            None
        """
        self.current_region = -1
        self.scroll()
        self.fig.draw()
        self.populate_tables()

    def marker_moved(self, comp: str, marker: str):
        """Updates plot and table after a marker has been moved."""
        if marker == Mark.ES.name:
            self.populate_tables()
            loc = self.data.markers.sel(
                quantity="frame",
                region=Region.GLOBAL.name,
                cine=self.cines_var.get(),
                comp=Comp.RAD.name,
                marker=Mark.ES.name,
            ).item()
            for f in self.fig.fixed_markers.values():
                f.set_xdata([loc, loc])
        else:
            self.update_table_one_marker(comp, marker)
            loc = (
                self.markers.sel(quantity="frame", comp=comp, marker=marker)
                .isel(region=self.current_region, missing_dims="ignore")
                .item()
            )
            self.fig.update_one_map(
                label=(comp, marker),
                images=self.images.sel(frame=loc),
                cylindrical=self.cylindrical.sel(frame=loc, comp=comp).isel(
                    region=self.current_region, missing_dims="ignore"
                ),
                draw=True,
            )

    def scroll(self, step=1, *args):
        """Changes the region being plotted when scrolling with the mouse."""
        current_region = (self.current_region + step) % self.regions

        if self.current_region != current_region:
            self.current_region = current_region
            self.fig.actions_manager.SimpleScroller.disabled = True
            self.fig.update_lines(
                self.velocities.isel(region=self.current_region, missing_dims="ignore"),
                draw=False,
            )
            self.fig.update_maps(
                self.images,
                self.cylindrical.isel(
                    region=self.current_region, missing_dims="ignore"
                ),
                self.markers.sel(quantity="frame").isel(
                    region=self.current_region, missing_dims="ignore"
                ),
                draw=False,
            )
            self.fig.update_markers(
                self.markers.sel(quantity="frame").isel(
                    region=self.current_region, missing_dims="ignore"
                ),
                draw=False,
            )
            self.populate_tables()
            self.fig.actions_manager.SimpleScroller.disabled = False

        return self.current_region, None, None

    def region_labels(self, size: int) -> Sequence:
        """Provides the region labels, if any.

        If the number of regions is 6, then these are named labels. Otherwise these are
        just numbered regions.

        Args:
            size (int): Number of regions to label.

        Returns:
            Sequence with the region labels.
        """
        if size == 6:
            labels = "AS", "A", "AL", "IL", "I", "IS"
            if self.data.orientation == "CCW":
                labels = labels[::-1]
            return labels
        else:
            return list(range(1, size + 1))

    def populate_tables(self, markers: Optional[xr.DataArray] = None) -> None:
        """Populates the information tables with the marker parameters.

        Args:
            markers (Optional, xr.DataArray): By default, this methods uses the markers
            of the currently selected cine and region, but if this argument is not None,
            it will use those markers instead.

        Returns:
            None
        """
        markers = markers if markers is not None else self.markers
        for t in self.param_tables.values():
            old_list = t.get_children()
            if len(old_list) > 0:
                t.delete(*old_list)

        labels = self.region_labels(markers.region.size)
        for comp, t in self.param_tables.items():
            vel = t.insert("", tk.END, text="Velocity (cm/s)", open=True)
            time = t.insert("", tk.END, text="Norm. Time (ms)", open=True)

            for i in range(markers.region.size):
                tag = "current" if i == self.current_region else "others"

                # We insert the velocities
                val = (
                    markers.sel(
                        marker=list(self.ids[comp]), comp=comp, quantity="velocity"
                    )
                    .isel(region=i, missing_dims="ignore")
                    .round(decimals=2)
                    .data.tolist()
                )
                t.insert(vel, tk.END, text=labels[i], values=val, tags=(tag,))

                # And the times
                val = (
                    markers.sel(marker=list(self.ids[comp]), comp=comp, quantity="time")
                    .isel(region=i, missing_dims="ignore")
                    .round(decimals=2)
                    .data.tolist()
                )
                t.insert(time, tk.END, text=labels[i], values=val, tags=(tag,))

    def update_table_one_marker(self, comp: str, marker: str):
        """Updates peak velocity and time table entry for a single marker."""
        t = self.param_tables[comp]

        # Update the velocity
        t.set(
            t.get_children(t.get_children()[0])[self.current_region],
            column=marker,
            value=self.markers.sel(comp=comp, marker=marker, quantity="velocity")
            .isel(region=self.current_region, missing_dims="ignore")
            .round(decimals=2)
            .item(),
        )

        # And the times
        t.set(
            t.get_children(t.get_children()[1])[self.current_region],
            column=marker,
            value=self.markers.sel(comp=comp, marker=marker, quantity="time")
            .isel(region=self.current_region, missing_dims="ignore")
            .round(decimals=2)
            .item(),
        )

    def update_velocities_list(self, cine):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities.sel(cine=cine)

        for v in self.velocities_frame.winfo_children():
            v.grid_remove()

        vel_list = set([v for v in velocities.region.data if "RADIAL" not in v])
        vel_list = sorted(vel_list, reverse=True)
        for i, v in enumerate(vel_list):
            ttk.Radiobutton(
                self.velocities_frame,
                text=v.lower().replace("_", " "),
                value=v,
                variable=self.velocities_var,
                command=self.replot,
            ).grid(row=0, column=i, sticky=tk.NSEW)

        self.velocities_var.set("GLOBAL")

    def calculate_velocities(self, cine, update=False):
        """Calculate pre-defined velocities for the chosen cine."""
        self.display_plots(False)
        self.controller.calculate_velocities(
            cine=cine,
            sign_reversal=tuple(-1 if var.get() else 1 for var in self.reverse_vel_var),
            update_velocities=update,
        )
        self.update_velocities_list(self.cines_var.get())

    def update_marker(self, marker, data, x, y, location):
        """When a marker moves, mask data should be updated."""
        self.controller.update_marker(
            markers=self.data.markers.sel(cine=self.cines_var.get()),
            marker_label=Mark[marker.get_label()],
            component=Comp[data.get_label().strip("_")],
            region=Region[self.vel_var],
            iregion=self.current_region,
            location=location,
            velocity_value=y,
            frames=self.velocities.sizes["frame"],
        )
        self.marker_moved(data.get_label().strip("_"), marker.get_label())

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
                filename=filename, cine=self.cines_var.get()
            )

    def export_superpixel(self, *args):
        """Exports the current superpixel velocity data to an XLSX file.

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
            self.reverse_vel_var[i].set(var.item() == -1)

    def update_widgets(self):
        """Updates widgets after an update in the data var."""
        self.populate_cine_box()
        self.update_sign_reversal()
        self.cine_changed()

    def clear_widgets(self):
        """Clear widgets after removing the data."""
        pass


def colour_figure(
    velocities: xr.DataArray,
    labels: Sequence,
    markers_idx: xr.DataArray,
    master: ttk.Frame,
) -> Figure:
    """Creates the color plots for the regional velocities.

    Args:
        velocities (xr.DataArray): Array with the velocities for the 24 angular regions.
        labels (Sequence): The region labels.
        markers_idx (xr.DataArray): Array with the frames of the global markers.
        master (ttk.Frame): PArent frame in which to place the figure.

    Returns:

    """
    fig = Figure(constrained_layout=True)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.NSEW)
    ax = fig.subplots(ncols=3, nrows=1)

    n_reg = velocities.sizes["region"]
    space = n_reg / len(labels)
    lines_pos = np.arange(space, n_reg, space) - 0.5
    labels_pos = np.arange(space // 2, n_reg, space) - 0.5

    for i, comp in enumerate((Comp.LONG.name, Comp.RAD.name, Comp.CIRC.name)):
        ax[i].imshow(
            velocities.sel(comp=comp),
            cmap=plt.get_cmap("jet"),
            aspect="auto",
            interpolation="bilinear",
        )
        ax[i].set_title(comp)

        ax[i].set_yticks(lines_pos, minor=True)
        ax[i].set_yticks(labels_pos, minor=False)
        ax[i].set_yticklabels(labels[::-1], minor=False)
        ax[i].yaxis.grid(True, which="minor", color="k", linestyle="-")
        ax[i].set_ylim((-0.5, n_reg - 0.5))

        idx = markers_idx.sel(comp=comp).dropna(dim="marker")
        ax[i].set_xticks(idx, minor=False)
        ax[i].set_xticklabels([m.item() for m in idx.marker], minor=False)

        fig.colorbar(ax[i].images[0], ax=ax[i], orientation="horizontal")

    fig.canvas.draw()
    return fig

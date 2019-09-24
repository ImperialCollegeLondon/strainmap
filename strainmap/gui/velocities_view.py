import tkinter as tk
from tkinter import ttk

from .base_window_and_task import Requisites, TaskViewBase, register_view


@register_view
class VelocitiesTaskView(TaskViewBase):

    requisites = Requisites.SEGMENTED

    def __init__(self, root):

        super().__init__(root, button_text="Velocities", button_image="speed.gif")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.datasets_box = None
        self.datasets_var = tk.StringVar(value="")
        self.velocities_frame = None
        self.velocities_var = tk.StringVar(value="")
        self.bg_method = (tk.BooleanVar(value=False), tk.BooleanVar(value=False))

        self.create_controls()

    def create_controls(self):
        """ Creates all the widgets of the view. """
        # Top frames
        control = ttk.Frame(master=self)
        visualise_frame = ttk.Frame(master=self)
        visualise_frame.columnconfigure(0, weight=1)
        visualise_frame.rowconfigure(0, weight=1)

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

        for i, method in enumerate(("Phantom", "Average")):
            ttk.Checkbutton(
                background_frame,
                text=method,
                variable=self.bg_method[i],
                command=self.on_bg_method_changed,
            ).grid(row=i, column=0, sticky=(tk.S, tk.W))

        # Grid all the widgets
        control.grid(sticky=tk.NSEW, padx=10, pady=10)
        visualise_frame.grid(sticky=tk.NSEW, padx=10, pady=10)
        dataset_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        self.datasets_box.grid(row=0, column=0, sticky=tk.NSEW)
        self.velocities_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=5, pady=5)
        add_velocity.grid(row=99, column=0, sticky=tk.NSEW)
        remove_velocity.grid(row=99, column=1, sticky=tk.NSEW)
        background_frame.grid(row=2, column=0, sticky=tk.NSEW, padx=5, pady=5)

    def dataset_changed(self, *args):
        """Updates the view when the selected dataset is changed."""
        current = self.datasets_var.get()
        self.update_velocities_list(current)

    def add_velocity(self):
        """Opens a dialog to add a new velocity to the list of velocities."""
        # result = AddVelocityDialog(self).show()
        # print(result)

    def remove_velocity(self):
        """Opens a dialog to remove the selected velocity to the list of velocities."""

    def update_velocities_list(self, dataset):
        """Updates the list of radio buttons with the currently available velocities."""
        velocities = self.data.velocities.get(dataset, {})

        for v in self.velocities_frame.winfo_children()[2:]:
            v.grid_remove()

        for i, v in enumerate(velocities):
            ttk.Radiobutton(
                self.velocities_frame, text=v, value=v, variable=self.velocities_var
            ).grid(row=i, column=0, columnspan=2, sticky=tk.NSEW)

        if self.velocities_var.get() not in velocities and len(velocities) > 0:
            self.velocities_var.set(list(velocities.keys())[0])

    def on_bg_method_changed(self):
        """Updates the velocities when the background substraction method changes."""

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        # Include only datasets with a segmentation
        values = list(self.data.segments.keys())
        current = self.datasets_var.get()
        self.datasets_box.config(values=values)
        if current not in values:
            current = values[0]
        self.datasets_var.set(current)

        # If there's already any velocity calculation, update the relevant widgets
        self.update_velocities_list(current)

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

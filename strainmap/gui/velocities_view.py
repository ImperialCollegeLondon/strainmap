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

        # Control-related attributes
        self.control = None

        # Visualization-related attributes
        self.visualise = None

        self.create_controls()

    def create_controls(self):
        """ Creates the control frame widgets. """
        self.control = ttk.Frame(master=self, name="control")
        self.control.grid(sticky=tk.NSEW, padx=10, pady=10)

        self.visualise = ttk.Frame(master=self, name="visualise")
        self.visualise.grid(sticky=tk.NSEW, padx=10, pady=10)

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        pass

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

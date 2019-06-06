import tkinter as tk
from tkinter import ttk

from ..base_classes import TaskBase, register_task, DataLoaded


@register_task
class Segmentation(TaskBase):

    pre_requisites = {DataLoaded}

    def __init__(self, root):

        super().__init__(root, button_text="Segmentation", button_image="molecules.gif")

    def create_controls(self):
        """ Creates the control frame widgets. """
        ttk.Label(self.control, text="The answer is 42.").grid(
            column=0, row=0, sticky=tk.NSEW, padx=5, pady=5
        )

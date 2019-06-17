import tkinter as tk
from tkinter import ttk

from .base_classes import Requisites, ViewBase, register_view


@register_view
class Segmentation(ViewBase):

    requisites = Requisites.DATALOADED

    def __init__(self, root, actions):

        super().__init__(
            root, actions, button_text="Segmentation", button_image="molecules.gif"
        )
        self.create_controls()

    def create_controls(self):
        """ Creates the control frame widgets. """
        ttk.Label(self.control, text="The answer is 42.").grid(
            column=0, row=0, sticky=tk.NSEW, padx=5, pady=5
        )

    def update_widgets(self):
        """ Updates widgets after an update in the data variable. """
        pass

    def clear_widgets(self):
        """ Clear widgets after removing the data. """
        pass

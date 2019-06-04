""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import tkinter as tk
from tkinter import ttk
from typing import Optional, Text

from .data_structures import StrainMapData
from .task_base import REGISTERED_TASKS
from .tasks import *  # noqa: F403,F401


class MainWindow(tk.Tk):
    """ StrainMap main window."""

    def __init__(self):
        super().__init__()
        self.title("StrainMap")
        self.minsize(1280, 720)

        self.rowconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.button_frame = ttk.Frame(self, width=150)
        self.control_frame = ttk.Frame(self, width=300)
        self.outputs_frame = ttk.Frame(self)

        self.button_frame.grid(column=0, row=1, sticky=tk.NS)
        self.control_frame.grid(column=1, row=1, sticky=tk.NS)
        self.outputs_frame.grid(column=2, row=1, sticky=tk.NSEW)

        self.button_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(0, weight=1)
        self.outputs_frame.columnconfigure(0, weight=1)
        self.control_frame.rowconfigure(0, weight=1)
        self.outputs_frame.rowconfigure(0, weight=1)
        self.control_frame.grid_propagate(flag=False)

        self.data = StrainMapData()
        self.loaded_tasks = {}
        self.achievements = set()

        self.unlock_achievement()

    def unlock_achievement(self, achievement: Optional[Text] = None):
        """ Adds achievements to the registry and runs the tasks loader. """
        if achievement is not None:
            self.achievements.add(achievement)

        for task in REGISTERED_TASKS - set(self.loaded_tasks.keys()):
            if len(task.pre_requisites - self.achievements) == 0:
                self.add_task(task)

    def lock_achievement(self, achievement: Text):
        """ Removes achievements from the registry and updates the loaded tasks. """
        self.achievements.discard(achievement)

        to_be_removed = []
        for task in self.loaded_tasks:
            if len(task.pre_requisites - self.achievements) > 0:
                self.loaded_tasks[task].unload()
                to_be_removed.append(task)

        # Finally we remove the unloaded tasks
        for task in to_be_removed:
            del self.loaded_tasks[task]

    def add_task(self, task):

        self.loaded_tasks[task] = task(root=self)

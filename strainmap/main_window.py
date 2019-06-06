""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import tkinter as tk
from tkinter import ttk
from typing import Optional, Text

from .base_classes import REGISTERED_TABS, REGISTERED_TASKS, REGISTERED_ACHIEVEMENTS
from .data_structures import StrainMapData
from .tasks_and_tabs import *  # noqa: F403,F401


class MainWindow(tk.Tk):
    """ StrainMap main window."""

    def __init__(self):
        super().__init__()
        self.title("StrainMap")
        self.minsize(1280, 720)
        self.protocol("WM_DELETE_WINDOW", self.__quit)
        self.closed = False

        self.rowconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.button_frame = ttk.Frame(self, width=150)
        self.control_frame = ttk.Frame(self, width=300)
        self.outputs_frame = ttk.Notebook(self)

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
        self.loaded_tabs = {}
        self.achievements = set()

        for a in REGISTERED_ACHIEVEMENTS:
            self.bind(a.LOCK, lambda _, ach=a: self.lock_achievement(ach))
            self.bind(a.UNLOCK, lambda _, ach=a: self.unlock_achievement(ach))

        self.unlock_achievement()

    def mainloop(self, *args):
        """ Finally, we initiate the main loop.

        This is a hack found here: http://github.com/matplotlib/matplotlib/issues/9637
        to avoid a crashing that happens when combining certain versions of tcl and
        certain versions of Python. It reveals when scrolling in a Matplotlib plot.
        It seems to be a problem only under MacOS.

        TODO: Investigate if there is a more elegant solution to this.
        """

        while not self.closed:
            try:
                self.update_idletasks()
                self.update()
            except UnicodeDecodeError:
                print("Caught Scroll Error")

    def __quit(self):
        """ Safe quit the program."""
        self.closed = True
        self.quit()

    def unlock_achievement(self, achievement=None):
        """ Adds achievements to the registry and runs the tasks loader. """
        if achievement is not None:
            self.achievements.add(achievement)

        for task in REGISTERED_TASKS - set(self.loaded_tasks.keys()):
            if len(task.pre_requisites - self.achievements) == 0:
                self.add_task(task)

        for tab in REGISTERED_TABS - set(self.loaded_tabs.keys()):
            if len(tab.pre_requisites - self.achievements) == 0:
                self.add_tab(tab)

    def lock_achievement(self, achievement):
        """ Removes achievements from the registry and updates loaded tasks and tabs."""
        self.achievements.discard(achievement)

        to_be_removed = []
        for task in self.loaded_tasks:
            if len(task.pre_requisites - self.achievements) > 0:
                self.loaded_tasks[task].unload()
                to_be_removed.append(task)

        # Finally we remove the unloaded tasks
        for task in to_be_removed:
            del self.loaded_tasks[task]

        to_be_removed = []
        for tab in self.loaded_tabs:
            if len(tab.pre_requisites - self.achievements) > 0:
                self.loaded_tabs[tab].unload()
                to_be_removed.append(tab)

        # Finally we remove the unloaded tabs
        for tab in to_be_removed:
            del self.loaded_tabs[tab]

    def add_task(self, task):
        self.loaded_tasks[task] = task(root=self)

    def add_tab(self, tab):
        self.loaded_tabs[tab] = tab(root=self)
        self.loaded_tabs[tab].create_tab_contents()

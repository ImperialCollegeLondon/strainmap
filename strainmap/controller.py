""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
import weakref

from .gui import *  # noqa: F403,F401
from .gui.base_window_and_task import (
    EVENTS,
    REGISTERED_BINDINGS,
    REGISTERED_TRIGGERS,
    REGISTERED_VIEWS,
    Requisites,
    bind_event,
)
from .models.strainmap_data_model import StrainMapData
from .models import quick_segmentation
from .models.velocities import calculate_velocities, update_marker
from .models.writers import velocity_to_xlsx


class StrainMap(object):
    """ StrainMap main window."""

    registered_views = REGISTERED_VIEWS

    def __init__(self):
        from .gui.base_window_and_task import MainWindow

        self.window = MainWindow()
        self.achieved = Requisites.NONE
        self.data = StrainMapData.from_folder()
        self.pair_events()
        self.unlock()

    def run(self):
        """ Runs StrainMap by calling the top window mainloop. """
        self.window.mainloop()

    def unlock(self, requisite=Requisites.NONE):
        """ Adds requisites and loads views. """
        self.achieved = self.achieved | requisite

        for view in self.registered_views:
            if Requisites.check(self.achieved, view.requisites):
                self.window.add(view, weakref.ref(self))

        self.update_views()

    def lock(self, requisite):
        """ Removes requisites and updates loaded views."""
        self.achieved = self.achieved ^ requisite

        for view in list(self.window.views):
            if not Requisites.check(self.achieved, view.requisites):
                self.window.remove(view)

    def update_views(self):
        """ Updates the data attribute of the views and the widgets depending on it. """
        for view in self.window.views:
            view.update_widgets()

    def pair_events(self):
        """ Pair the registered triggers with the registered binds.

        This will produce an error if there is a trigger without an associated binding.
        """
        for ev in REGISTERED_TRIGGERS:
            assert ev in REGISTERED_BINDINGS.keys()
            EVENTS[ev] = lambda control=self, event=ev, **kwargs: REGISTERED_BINDINGS[
                event
            ](control, **kwargs)

    @bind_event
    def load_data_from_folder(self, view, data_files):
        """Creates a StrainMapData object."""
        self.data = StrainMapData.from_folder(data_files)

        if self.data.data_files:
            self.unlock(Requisites.DATALOADED)
        else:
            self.lock(Requisites.DATALOADED)
        view.update_widgets()

    @bind_event
    def load_data_from_file(self, view, strainmap_file):
        """Creates a StrainMapData object."""
        self.data = StrainMapData.from_file(strainmap_file)

        if self.data.data_files:
            self.unlock(Requisites.DATALOADED)
        else:
            self.lock(Requisites.DATALOADED)

        if any(len(i) != 0 for i in self.data.segments.values()):
            self.unlock(Requisites.SEGMENTED)
        else:
            self.lock(Requisites.SEGMENTED)

        view.update_widgets()

    @bind_event
    def clear_data(self, view, **kwargs):
        """Clears the StrainMapData object from the widgets."""
        if kwargs.get("clear", False):
            self.data = StrainMapData.from_folder()
            self.lock(Requisites.DATALOADED)
            self.lock(Requisites.SEGMENTED)
            view.update_widgets()

    @bind_event
    def add_paths(self, view, data_files=None, bg_files=None):
        if self.data.add_paths(data_files, bg_files):
            view.update_widgets()

    @bind_event
    def add_h5_file(self, view, strainmap_file):
        if self.data.add_h5_file(strainmap_file):
            view.update_widgets()

    @bind_event
    def find_segmentation(self, view, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.find_segmentation(data=self.data, **kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        view.update_widgets()

    @bind_event
    def update_segmentation(self, view, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.update_segmentation(data=self.data, **kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        view.update_widgets()

    @bind_event
    def update_and_find_next(self, view, **kwargs):
        """Runs an automated segmentation routine."""
        quick_segmentation.update_and_find_next(data=self.data, **kwargs)
        view.update_widgets()

    @bind_event
    def clear_segmentation(self, view, **kwargs):
        """Clears an existing segmentation."""
        quick_segmentation.clear_segmentation(data=self.data, **kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        view.update_widgets()

    @bind_event
    def calculate_velocities(self, view, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        calculate_velocities(data=self.data, **kwargs)
        view.update_widgets()

    @bind_event
    def update_marker(self, view, **kwargs):
        """Updates the markers information after moving one of them."""
        update_marker(data=self.data, **kwargs)
        view.update_widgets()

    @bind_event
    def export_velocity(self, **kwargs):
        """Exports velocity data to a XLSX file."""
        velocity_to_xlsx(data=self.data, **kwargs)

""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
from importlib import reload

from .gui import *  # noqa: F403,F401
from .gui.base_window_and_task import (
    EVENTS,
    REGISTERED_BINDINGS,
    REGISTERED_TRIGGERS,
    REGISTERED_VIEWS,
    Requisites,
    bind_event,
)
from .models.strainmap_data_model import factory
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
        self.data = None
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
                self.window.add(view)
                self.window.views[-1].data = self.data

    def lock(self, requisite):
        """ Removes requisites and updates loaded views."""
        self.achieved = self.achieved ^ requisite

        for view in list(self.window.views):
            if not Requisites.check(self.achieved, view.requisites):
                self.window.remove(view)

    def update_views(self):
        """ Updates the data attribute of the views and the widgets depending on it. """
        for view in self.window.views:
            view.data = self.data

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
    def load_data(self, **kwargs):
        """Creates a StrainMapData object."""
        self.data = factory(**kwargs)

        if self.data.data_files:
            self.unlock(Requisites.DATALOADED)
        else:
            self.lock(Requisites.DATALOADED)

        if any(len(i) != 0 for i in self.data.segments.values()):
            self.unlock(Requisites.SEGMENTED)
        else:
            self.lock(Requisites.SEGMENTED)
        self.update_views()

    @bind_event
    def clear_data(self, **kwargs):
        """Clears the StrainMapData object from the widgets."""
        if kwargs.get("clear", False):
            self.data = None
            self.lock(Requisites.DATALOADED)
            self.lock(Requisites.SEGMENTED)
            self.update_views()

    @bind_event
    def find_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        reload(quick_segmentation)
        self.data = quick_segmentation.find_segmentation(**kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        self.update_views()

    @bind_event
    def update_segmentation(self, unlock=True, **kwargs):
        """Runs an automated segmentation routine."""
        self.data = quick_segmentation.update_segmentation(**kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if there_are_segments and unlock:
            self.unlock(Requisites.SEGMENTED)
        elif not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        self.update_views()

    @bind_event
    def update_and_find_next(self, **kwargs):
        """Runs an automated segmentation routine."""
        self.data = quick_segmentation.update_and_find_next(**kwargs)
        self.update_views()

    @bind_event
    def clear_segmentation(self, **kwargs):
        """Clears an existing segmentation."""
        self.data = quick_segmentation.clear_segmentation(**kwargs)
        there_are_segments = any(len(i) != 0 for i in self.data.segments.values())
        if not there_are_segments:
            self.lock(Requisites.SEGMENTED)
        self.update_views()

    @bind_event
    def calculate_velocities(self, **kwargs):
        """Calculates the velocities based on a given segmentation."""
        self.data = calculate_velocities(**kwargs)
        self.update_views()

    @bind_event
    def update_marker(self, **kwargs):
        """Updates the markers information after moving one of them."""
        self.data = update_marker(**kwargs)
        self.update_views()

    @bind_event
    def export_velocity(self, **kwargs):
        """Exports velocity data to a XLSX file."""
        velocity_to_xlsx(**kwargs)

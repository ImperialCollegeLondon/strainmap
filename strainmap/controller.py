""" Entry point of StrainMap, creating the main window and variables used along the
whole code. """
from .gui.base_window_and_task import (
    REGISTERED_VIEWS,
    REGISTERED_BINDINGS,
    REGISTERED_TRIGGERS,
    EVENTS,
    Requisites,
    bind_event,
)
from .models.strainmap_data_model import factory
from .gui import *  # noqa: F403,F401


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

    def update_views(self, data):
        """ Updates the data attribute of the views and the widgets depending on it. """
        self.data = data
        for view in self.window.views:
            view.data = data

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
        """ Creates a StrainMapData object. """
        if kwargs:
            data = factory(**kwargs)
            self.unlock(Requisites.DATALOADED)
            self.update_views(data)

    @bind_event
    def clear_data(self, **kwargs):
        """ Clears the StrainMapData object from the widgets. """
        if kwargs.get("clear", False):
            self.lock(Requisites.DATALOADED)
            self.update_views(None)

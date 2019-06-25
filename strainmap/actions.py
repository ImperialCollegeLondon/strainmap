"""This module contains the functions that interface together the GUI, the
model and the main control object.

All this functions take as first argument a StrainMap object, which will update the
data available in the views, and none or more keyword arguments that are the actual
inputs to the model functions.
"""
from typing import Callable

from .gui.base_classes import Requisites
from .model.strainmap_data_model import factory

REGISTERED_ACTIONS = {}
""" Registered actions."""


def register_action(fun: Callable):
    """Registers an action that can be executed by the views."""

    REGISTERED_ACTIONS[fun.__name__] = fun

    return fun


@register_action
def load_data(control, **kwargs):
    """Creates a StrainMapData object."""
    data = factory(**kwargs)
    control.unlock(Requisites.DATALOADED)
    control.update_views(data)


@register_action
def clear_data(control, **kwargs):
    """Clears the StrainMapData object from the widgets."""
    control.lock(Requisites.DATALOADED)
    control.update_views(None)

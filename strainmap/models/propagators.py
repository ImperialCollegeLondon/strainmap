import numpy as np

from .contour_mask import Contour

REGISTERED_PROPAGATORS: dict = {}
""" Dictionary with all the propagators available in StrainMap."""


def register_propagator(fun):
    """Register a propagator, so it is available across StrainMap."""

    msg = f"Propagator {fun.__name__} already exists."
    assert fun.__name__ not in REGISTERED_PROPAGATORS, msg

    REGISTERED_PROPAGATORS[fun.__name__] = fun

    return fun


@register_propagator
def initial(*, initial, **kwargs) -> Contour:
    """There is no propagation: always use the same initial contour."""
    return initial


@register_propagator
def previous(*, previous, **kwargs) -> Contour:
    """The next initial is equal to the previously calculated one.

    Optionally, a 'dilation_factor' can be included to contract/expand the contour.
    """
    if previous is None:
        raise RuntimeError("'previous' cannot be None in the 'previous' propagator.")

    dilation = kwargs.get("dilation_factor", None) if kwargs else None
    return previous if not dilation else previous.dilate(p=dilation)


@register_propagator
def weighted(*, initial, previous, **kwargs) -> Contour:
    """The next initial is a weighted average between the previous and the initial.

    The relative weight is given by the keyword 'weight', with 1 resulting in the
    initial contour and 0 resulting in the previous one.
    """
    from copy import copy

    if previous is None or initial is None:
        msg = "'initial' and 'previous' cannot be None in the 'weighted' propagator."
        raise RuntimeError(msg)

    w = np.clip(kwargs.get("weight", 0), 0, 1) if kwargs else 0
    out = copy(previous)

    ir = np.interp(previous.polar.theta, initial.polar.theta, initial.polar.r)
    r = w * ir + (1 - w) * previous.polar.r

    out.polar = np.array([r, previous.polar.theta]).T

    return out

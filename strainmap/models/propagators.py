from typing import Optional, Callable, Tuple
import numpy as np

from .contour_mask import Contour


PROPAGATORS: dict = {}
""" Dictionary with all the propagators available in StrainMap."""

PROPAGATORS_SIGNATURE = Callable[
    [Optional[Contour], Optional[Contour], Optional[dict]], Contour
]
"""Propagators signature."""

COMBINED_PROPAGATORS_SIGNATURE = Callable[
    [Tuple[Contour, Contour], Tuple[Contour, Contour], Optional[dict]],
    Tuple[Contour, Contour],
]
"""Combined propagators signature."""


def register_propagator(fun: PROPAGATORS_SIGNATURE):
    """ Register a propagator, so it is available across StrainMap. """

    msg = f"Propagator named {fun.__name__} already exists."
    assert fun.__name__ not in PROPAGATORS, msg

    PROPAGATORS[fun.__name__] = fun

    return fun


@register_propagator
def initial(
    initial: Optional[Contour] = None,
    previous: Optional[Contour] = None,
    options: Optional[dict] = None,
) -> Contour:
    """ There is no propagation: always use the same initial contour. """
    if initial is None:
        raise RuntimeError("'initial' cannot be None in the 'initial' propagator.")

    return initial


@register_propagator
def previous(
    initial: Optional[Contour] = None,
    previous: Optional[Contour] = None,
    options: Optional[dict] = None,
) -> Contour:
    """ The next initial is equal to the previously calculated one.

    Optionally, a 'dilation_factor' can be included to contract/expand the contour.
    """
    if previous is None:
        raise RuntimeError("'previous' cannot be None in the 'previous' propagator.")

    dilation = options.get("dilation_factor", None) if options else None
    return previous if not dilation else previous.dilate(p=dilation)


@register_propagator
def weighted(
    initial: Optional[Contour] = None,
    previous: Optional[Contour] = None,
    options: Optional[dict] = None,
) -> Contour:
    """ The next initial is a weighted average between the previous and the initial.

    The relative weight is given by the keyword 'weight', with 1 resulting in the
    initial contour and 0 resulting in the previous one.
    """
    if previous is None or initial is None:
        msg = "'initial' and 'previous' cannot be None in the 'weighted' propagator."
        raise RuntimeError(msg)

    w = np.clip(options.get("weight", 0), 0, 1) if options else 0
    out = previous.to_contour()

    irho = np.interp(previous.polar[:, 1], initial.polar[:, 1], initial.polar[:, 0])
    rho = w * irho + (1 - w) * previous.polar[:, 0]

    out.polar = np.array([rho, previous.polar[:, 1]]).T

    return out

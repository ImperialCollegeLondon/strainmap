from typing import Text, Callable

from skimage.filters import gaussian
from skimage.segmentation import inverse_gaussian_gradient


REGISTERED_FILTERS = {
    "gaussian": gaussian,
    "inverse_gaussian": inverse_gaussian_gradient,
}
""" Registered filters. """


def get_filter(name: Text) -> Callable:
    """Returns the callable of the chosen filter model."""
    try:
        return REGISTERED_FILTERS[name]
    except KeyError:
        raise KeyError(f"Filter {name} does not exists!")

from skimage.filters import gaussian
from skimage.segmentation import inverse_gaussian_gradient

REGISTERED_FILTERS = {
    "gaussian": gaussian,
    "inverse_gaussian": inverse_gaussian_gradient,
}
""" Registered filters. """

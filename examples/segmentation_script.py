"""This script contains a simple example on how to run a segmentation of one or more
images providing the initial contour and setting the segmentation model, filter,
propagator and their corresponding input parameters.

Installation instructions on how to have StrainMap up and running are included in the
README.md file of the StrainMap repository.

To run this script simply activate the virtual environment created in the installation
instructions and then run:

$ python path_to_this_script/segmentation_script.py

Full details of the inputs of the 'simple_segmentation' function can be obtained with:

>>> from strainmap.models.quick_segmentation import simple_segmentation
>>> help(simple_segmentation)

The available parameters for the model and filter functions - as well as their default
values - can be found in:

https://scikit-image.org/docs/dev/api/skimage.segmentation.html
https://scikit-image.org/docs/dev/api/skimage.filters.html
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pydicom

from strainmap.models.contour_mask import Contour
from strainmap.models.segmentation import _simple_segmentation

# Load the data of interest
path = Path(__file__).parent.parent / "tests/data/SUB1/MR012001002.dcm"
image = pydicom.dcmread(str(path)).pixel_array

# Create the initial contour
init_out = Contour.circle(center=(305, 275), radius=60, shape=image.shape).xy
init_in = Contour.circle(center=(310, 280), radius=40, shape=image.shape).xy

# Choose the models and parameters. See 'simple_segmentation' for a description of
# all the options
model = "AC"
ffilter = "gaussian"
propagator = "initial"
model_params = dict(alpha=0.0098, beta=15.5, gamma=0.0022)
filter_params = dict(sigma=2)
propagator_params = dict()

# Launch the segmentation process
outer = _simple_segmentation(
    data=image,
    initial=init_out,
    model=model,
    ffilter=ffilter,
    propagator=propagator,
    model_params=model_params,
    filter_params=filter_params,
    propagator_params=propagator_params,
)

inner = _simple_segmentation(
    data=image,
    initial=init_in,
    model=model,
    ffilter=ffilter,
    propagator=propagator,
    model_params=model_params,
    filter_params=filter_params,
    propagator_params=propagator_params,
)

# Plot the results
plt.imshow(image)
plt.plot(*init_out.T, "r", label="Initial Out")
plt.plot(*init_in.T, "r--", label="Initial In")
plt.plot(*outer, "y", label="Segmented Outer")
plt.plot(*inner, "y--", label="Segmented Inner")

plt.legend()
plt.show()

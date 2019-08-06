from pathlib import Path

import matplotlib.pyplot as plt
import pydicom

from strainmap.models.contour_mask import Contour
from strainmap.models.quick_segmentation import simple_segmentation

# Load the data of interest
path = Path(__file__).parent.parent / "tests/data/SUB1/MR024001000.dcm"
image = pydicom.dcmread(str(path)).pixel_array

# Create the initial contour
init = Contour.circle((310, 280), radius=60, shape=image.shape).xy

# Choose the models and parameters. See 'simple_segmentation' for a description of
# all the options
model = "AC"
ffilter = "gaussian"
propagator = "initial"
model_params = dict(alpha=0.01, beta=10, gamma=0.002)
filter_params = dict(sigma=1)
propagator_params = dict()

# Launch the segmentation process
segment = simple_segmentation(
    data=image,
    initial=init,
    model=model,
    ffilter=ffilter,
    propagator=propagator,
    model_params=model_params,
    filter_params=filter_params,
    propagator_params=propagator_params,
)

# Plot the results
plt.imshow(image)
plt.plot(init[:, 0], init[:, 1], "r", label="Initial")
plt.plot(segment[:, 0], segment[:, 1], "y", label="Segmented")
plt.legend()
plt.show()

import numpy as np
import xarray as xr


def test_dict_to_xarray():
    import numpy as np
    from strainmap.models.transformations import dict_to_xarray

    shape = (3, 2, 3, 2, 4)
    dims = ["Character", "Equipment", "points 1", "points 2", "points 3"]
    structure = {
        "Frodo": {
            "Sword": np.random.random(shape[2:]),
            "Shield": np.random.random(shape[2:]),
        },
        "Sam": {
            "Sword": np.random.random(shape[2:]),
            "Shield": np.random.random(shape[2:]),
        },
        "Merry": {
            "Sword": np.random.random(shape[2:]),
            "Shield": np.random.random(shape[2:]),
        },
    }
    actual = dict_to_xarray(structure, dims)
    assert actual.shape == shape
    assert all(actual.Character.data == list(structure.keys()))
    assert all(actual.Equipment.data == list(structure["Frodo"].keys()))


def test_masked_reduction(
    radial_mask,
    angular_mask,
    expanded_radial,
    expanded_angular,
    reduced_radial,
    reduced_angular,
):
    from strainmap.models.transformations import masked_reduction

    # Masks are reduced, as it will be the real case, covering only certain ROI
    radial = radial_mask.sel(row=radial_mask.row[1:-1], col=radial_mask.col[1:-1])
    angular = angular_mask.sel(row=angular_mask.row[1:-1], col=angular_mask.col[1:-1])

    # An input array with radial symmetry in the row/col plane should return an array
    #  with no angular dependence
    actual = masked_reduction(expanded_radial, radial, angular)
    xr.testing.assert_equal(actual, reduced_radial)

    # Likewise, if input has angular symmetry, the return value should have no radial
    #  dependence
    actual = masked_reduction(expanded_angular, radial, angular)
    np.testing.assert_equal(actual.data, reduced_angular)


def test_masked_expansion(
    radial_mask,
    angular_mask,
    expanded_radial,
    expanded_angular,
    reduced_radial,
    reduced_angular,
):
    from strainmap.models.transformations import masked_expansion

    nrow = angular_mask.sizes["row"]
    ncol = angular_mask.sizes["col"]

    # Masks are reduced, as it will be the real case, covering only certain ROI
    radial = radial_mask.sel(row=radial_mask.row[1:-1], col=radial_mask.col[1:-1])
    angular = angular_mask.sel(row=angular_mask.row[1:-1], col=angular_mask.col[1:-1])

    # An input array with no angular dependence should produce an output array with
    # radial symmetry in the row/col plane.
    actual = masked_expansion(reduced_radial, radial, angular, nrow, ncol)
    xr.testing.assert_equal(actual, expanded_radial.where(~actual.isnull()))

    # Likewise, an input array no radial dependence should produce an output array with
    # angular symmetry in the row/col plane.
    actual = masked_expansion(reduced_angular, radial, angular, nrow, ncol)
    xr.testing.assert_equal(actual, expanded_angular.where(~actual.isnull()))

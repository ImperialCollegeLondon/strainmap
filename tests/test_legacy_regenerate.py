def test_dict_to_xarray():
    import numpy as np
    from strainmap.models.legacy_regenerate import _dict_to_xarray

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
    actual = _dict_to_xarray(structure, dims)
    assert actual.shape == shape
    assert all(actual.Character.data == list(structure.keys()))
    assert all(actual.Equipment.data == list(structure["Frodo"].keys()))

from pytest import raises, mark


def test_regional_labels():
    from collections.abc import Hashable
    from strainmap.models.sm_data import RegionalLabel, Region

    a = RegionalLabel(Region.ANGULAR, 6)
    b = RegionalLabel(Region.ANGULAR, 7)
    c = RegionalLabel.from_str(b.as_str)

    assert a != b
    assert c == b
    assert hash(c) == hash(c.as_str) == hash(b)
    assert isinstance(a, Hashable)


def test_labelled_array_creation(larray):
    assert len(larray.dims) == 3
    assert len(larray.coords) == 3

    for k, v in larray.coords.items():
        idx = larray.dims.index(k)
        assert v is None or len(v) == larray.values.shape[idx]


def test_creation_from_dict():
    from strainmap.models.sm_data import LabelledArray
    import numpy as np

    key1 = ["dog", "cat", "mouse"]
    key2 = ["big", "medium", "tiny", "micro"]
    var1 = ["a", "b", "c"]

    nested_dict = {k1: {k2: np.random.random((3, 2, 5)) for k2 in key2} for k1 in key1}

    result = LabelledArray.from_dict(
        dims=["animal", "size", "var1", "var2", "var3"],
        coords={"var1": var1},
        values=nested_dict,
        skip=("mouse",),
    )

    assert result.shape == (2, 4, 3, 2, 5)
    assert result.coords["animal"] == ["dog", "cat"]
    assert result.coords["size"] == key2
    assert result.coords["var1"] == var1
    assert result.coords["var2"] is None and result.coords["var3"] is None


def test_len_of(larray):
    for d, i in zip(larray.dims, larray.shape):
        assert larray.len_of(d) == i


def test_labelled_array_selection(larray):
    small = larray[0]
    assert len(small.dims) == 2
    assert small.shape == (larray.shape[1:])

    small = larray[:, 0]
    assert len(small.dims) == 2
    assert small.shape == (larray.shape[0:3:2])

    small2 = larray.sel(cols="x")
    assert small == small2

    small4 = larray.sel(cols="y")
    assert len(small4.dims) == 3
    assert len(small4.coords["cols"]) == 2
    assert all([c == "y" for c in small4.coords["cols"]])
    assert small4.shape == (3, 2, 4)


def test_transpose_labelled_array(larray):
    new = larray.transpose("depth", "rows", "cols")
    assert new.shape == (4, 3, 3)


def test_sum(larray):
    from numpy.random import choice

    assert isinstance(larray.sum(), float)

    dims = tuple(choice(larray.dims, size=2, replace=False))
    didx = tuple((larray.dims.index(i) for i in dims))
    summed = larray.sum(dims=dims)
    expected_values = larray.values.sum(axis=didx)

    assert not any([d in summed.dims for d in dims])
    assert not any([d in summed.coords for d in dims])
    assert (summed.values == expected_values).any()


def test_mean(larray):
    from numpy.random import choice

    assert isinstance(larray.mean(), float)

    dims = tuple(choice(larray.dims, size=2, replace=False))
    summed = larray.mean(dims=dims)

    assert not any([d in summed.dims for d in dims])
    assert not any([d in summed.coords for d in dims])


def test_cumsum_coo(larray_coo):
    with raises(NotImplementedError):
        larray_coo.cumsum()


def test_cumsum_np(larray_np):
    from numpy.random import choice

    assert isinstance(larray_np.cumsum(), type(larray_np.values))

    dim = choice(larray_np.dims)
    summed = larray_np.cumsum(dim=dim)
    assert summed.dims == larray_np.dims
    assert summed.coords == larray_np.coords
    assert summed.shape == larray_np.shape


def test_concatenate(larray):
    from numpy.random import choice, randint
    from copy import deepcopy
    from strainmap.models.sm_data import LabelledArray

    dim = choice(larray.dims)
    didx = larray.dims.index(dim)
    N = randint(1, 5)
    joint = larray.concatenate([larray] * N, dim=dim)

    assert joint.dims == larray.dims
    for d, c in joint.coords.items():
        if d == dim:
            if c is None:
                assert c == larray.coords[d]
            else:
                assert len(c) == (N + 1) * len(larray.coords[d])
            assert joint.shape[didx] == (N + 1) * larray.shape[didx]
        else:
            assert c == larray.coords[d]
            assert (
                joint.shape[larray.dims.index(d)] == larray.shape[larray.dims.index(d)]
            )

    with raises(KeyError):
        larray.concatenate([larray], dim="cat")

    with raises(TypeError):
        new_coords = deepcopy(larray.coords)
        new_coords[dim] = (
            None if larray.coords[dim] else list(range(larray.shape[didx]))
        )
        new = LabelledArray(larray.dims, new_coords, larray.values)
        larray.concatenate([new], dim=dim)


@mark.parametrize("coords", [None, ["maine coon"] * 5])
def test_stack(larray, coords):
    from copy import deepcopy
    from strainmap.models.sm_data import LabelledArray

    dim = "cat"
    joint = larray.stack([larray] * 4, dim=dim, coords=coords)

    assert dim in joint.dims
    assert dim in joint.coords
    assert joint.coords[dim] is coords
    assert joint.shape[0] == 5

    with raises(ValueError):
        new_coords = deepcopy(larray.coords)
        new_coords[larray.dims[0]] = (
            None if larray.coords[larray.dims[0]] else list(range(larray.shape[0]))
        )
        new = LabelledArray(larray.dims, new_coords, larray.values)
        larray.stack([new], dim=dim)

    with raises(ValueError):
        new_dims = ["dog"] + list(larray.dims[1:])
        new_coords = {d: c for d, c in larray.coords.items() if d != larray.dims[0]}
        new_coords["dog"] = larray.coords[larray.dims[0]]
        new = LabelledArray(new_dims, new_coords, larray.values)
        larray.stack([new], dim=dim)

    with raises(ValueError):
        larray.stack([larray], dim=dim, coords=["7"])


def test_align(larray):
    import numpy as np

    # Same dimensions, different order
    new_dims = list(larray.dims)
    np.random.shuffle(new_dims)
    rightin = larray.transpose(*new_dims)
    left, right = larray.align(rightin)
    assert left.dims == right.dims
    assert all(d in new_dims for d in left.dims)
    assert all(larray.len_of(d) == left.len_of(d) for d in larray.dims)
    assert all(larray.len_of(d) == right.len_of(d) for d in larray.dims)

    # Some common dimensions
    leftin = larray.sel(cols="x")
    rightin = larray.sel(depth="shallow")
    left, right = leftin.align(rightin)
    assert left.dims == right.dims
    assert all(v is None for k, v in left.coords.items() if k not in leftin.dims)
    assert all(v is None for k, v in right.coords.items() if k not in rightin.dims)
    assert all(
        left.len_of(d) == leftin.len_of(d) if d in leftin.dims else left.len_of(d) == 1
        for d in left.dims
    )
    assert all(
        right.len_of(d) == rightin.len_of(d)
        if d in rightin.dims
        else right.len_of(d) == 1
        for d in right.dims
    )

    # No common dimensions
    leftin = larray.sel(cols="x", rows=1)
    rightin = larray.sel(depth="shallow")
    left, right = leftin.align(rightin)
    assert left.dims == right.dims
    assert all(v is None for k, v in left.coords.items() if k in rightin.dims)
    assert all(v is None for k, v in right.coords.items() if k in leftin.dims)
    assert all(
        left.len_of(d) == leftin.len_of(d) if d in leftin.dims else left.len_of(d) == 1
        for d in left.dims
    )
    assert all(
        right.len_of(d) == rightin.len_of(d)
        if d in rightin.dims
        else right.len_of(d) == 1
        for d in right.dims
    )

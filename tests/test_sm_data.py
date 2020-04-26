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


def test_labelled_array_selection(larray):
    small = larray[:, 0]
    assert len(small.dims) == 2
    assert small.shape == (larray.shape[0:3:2])

    small2 = larray.sel(cols="x")
    small3 = larray.sel(cols__i=0)
    assert small == small2
    assert small == small3

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
    summed = larray.sum(dims=dims)

    assert not any([d in summed.dims for d in dims])
    assert not any([d in summed.coords for d in dims])


def test_mean(larray):
    from numpy.random import choice

    assert isinstance(larray.mean(), float)

    dims = tuple(choice(larray.dims, size=2, replace=False))
    summed = larray.mean(dims=dims)

    assert not any([d in summed.dims for d in dims])
    assert not any([d in summed.coords for d in dims])


def test_cumsum(larray):
    from numpy.random import choice
    import sparse

    if isinstance(larray.values, sparse.COO):
        with raises(NotImplementedError):
            larray.cumsum()
    else:
        assert isinstance(larray.cumsum(), type(larray.values))

        dim = choice(larray.dims)
        summed = larray.cumsum(dim=dim)
        assert summed.dims == larray.dims
        assert summed.coords == larray.coords
        assert summed.shape == larray.shape


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

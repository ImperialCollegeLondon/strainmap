import numpy as np
from pytest import approx, mark


def test_inplane_strain(data_with_velocities):
    from strainmap.models.strain import calculate_inplane_strain

    strain = calculate_inplane_strain(
        data_with_velocities, datasets=data_with_velocities.data_files.datasets[:1]
    )
    assert set(strain) == set(data_with_velocities.data_files.datasets[:1])
    assert strain[data_with_velocities.data_files.datasets[0]].shape == (2, 3, 512, 512)


@mark.parametrize("deltaz", [1, 1.2])
def test_outofplane_strain(deltaz):
    from strainmap.models.strain import calculate_outofplane_strain
    from types import SimpleNamespace

    def vel(x, y, z, t):
        return [0, 0, x % 7 + 2 * y - 3 * z + 4 * t]

    data = SimpleNamespace()
    data.masks = {
        f"dodo{z}": {
            "cylindrical - Estimated": np.transpose(
                [
                    [[vel(x, y, z, t) for y in range(100)] for x in range(100)]
                    for t in range(10)
                ],
                (3, 0, 1, 2),
            ),
            "angular x6 - Estimated": np.repeat(
                np.arange(100, dtype=int)[:, None] % 7, 100, 1
            ),
        }
        for z in range(8)
    }
    data.data_files = SimpleNamespace()
    data.data_files.files = list(data.masks.keys())
    data.data_files.slice_loc = lambda x: {
        k: deltaz * i for i, k in enumerate(data.data_files.files)
    }[x]

    strain = calculate_outofplane_strain(
        data, image_axes=(-2, -1), component_axis=0  # type: ignore
    )
    assert strain == approx(-3 / deltaz)


@mark.parametrize("masked", [False, True])
def test_inplane_strain_rate(masked):
    from .conftest import vector_field
    from strainmap.models.strain import inplane_strain_rate
    import numpy as np

    r0 = np.random.rand() + 0.1
    t0 = np.random.rand() + 0.1
    xsize = ysize = np.random.randint(21, 50)
    origin = np.array((xsize, ysize)) / 2
    px, py = np.mgrid[range(xsize), range(ysize)]
    inner = np.sqrt((px - origin[1]) ** 2 + (py - origin[0]) ** 2) > xsize / 7
    outer = np.sqrt((px - origin[1]) ** 2 + (py - origin[0]) ** 2) < xsize / 2.5
    mask = np.tile(~np.bitwise_and(inner, outer), (3, 1, 1))

    # Radial field
    velocities = vector_field(r0=r0, xsize=xsize, ysize=ysize)[0]
    if masked:
        velocities = np.ma.array(velocities, mask=mask)
    pradial, pazimutal = inplane_strain_rate(velocities[:2, 0], origin=origin)
    assert np.mean(pradial) == approx(r0, rel=1e-1)
    assert np.mean(pazimutal) == approx(r0, rel=1e-1)

    # Angular field
    velocities = vector_field(t0=t0, xsize=xsize, ysize=ysize)[0]
    if masked:
        velocities = np.ma.array(velocities, mask=mask)
    pradial, pazimutal = inplane_strain_rate(velocities[:2, 0], origin=origin)
    assert np.mean(pradial) == approx(0, rel=1e-2)
    az = np.nan_to_num(pazimutal)[int(xsize / 2)]
    lcentre = int(ysize / 2) - 2
    rcentre = int(ysize / 2) + 2
    assert (az[:lcentre] >= 0).all()
    assert (np.diff(az[:lcentre]) >= 0).all()
    assert (az[rcentre:] <= 0).all()
    assert (np.diff(az[rcentre:]) >= 0).all()

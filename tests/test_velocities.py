from _pytest.python_api import approx


def test_find_theta0():
    from strainmap.models.velocities import find_theta0
    import numpy as np

    N = 5
    expected = np.random.uniform(-np.pi, np.pi, N)

    vector = np.zeros((N, 2, 2))
    vector[:, 1, 0] = np.sin(expected)
    vector[:, 0, 0] = np.cos(expected)

    actual = find_theta0(vector)
    assert actual == approx(expected)


def test_scale_phase(strainmap_data):
    from strainmap.models.velocities import scale_phase

    dataset_name = list(strainmap_data.data_files.keys())[0]
    phase = scale_phase(strainmap_data, dataset_name)

    assert (phase >= -0.5).all()
    assert (phase <= 0.5).all()


def test_global_masks_and_origin():
    from strainmap.models.velocities import global_masks_and_origin
    from strainmap.models.contour_mask import Contour
    import numpy as np

    centre = (11, 11)
    shape = (21, 21)
    c1 = Contour.circle(centre, 6, shape=shape)
    c2 = Contour.circle(centre, 4, shape=shape)

    mask, origin = global_masks_and_origin([c1.xy.T], [c2.xy.T], shape)

    assert len(mask) == 1
    assert mask[0].shape == shape
    assert len(origin) == 1
    assert origin[0] == approx(np.array(centre), abs=0.5)


def test_transform_to_cylindrical():
    from strainmap.models.velocities import transform_to_cylindrical
    from strainmap.models.contour_mask import cylindrical_projection
    import numpy as np

    cartvel = np.random.random((3, 5, 48, 58))
    masks = np.random.randint(0, 2, (5, 48, 58))
    origin = np.array([48] * 10).reshape((5, 2))

    velocities = transform_to_cylindrical(cartvel, masks, origin)

    assert velocities.shape == cartvel.shape
    assert velocities[2] == approx(cartvel[2])

    num = np.sum(masks[None, :, :, :], axis=(2, 3))
    bulk = np.sum(cartvel * masks[None, :, :, :], axis=(2, 3)) / num
    bulk[-1] = 0

    expected = np.zeros_like(cartvel)
    for i in range(5):
        expected[:, i, :, :] = cylindrical_projection(
            cartvel[:, i, :, :] - bulk[:, i, None, None],
            origin[i],
            component_axis=0,
            image_axes=(1, 2),
        )

    assert expected == approx(velocities)


def test_calculate_velocities(segmented_data):
    from strainmap.models.velocities import calculate_velocities

    dataset_name = list(segmented_data.data_files.keys())[0]

    velocities = calculate_velocities(
        segmented_data,
        dataset_name,
        global_velocity=True,
        angular_regions=[6],
        # radial_regions=[4],
    ).velocities

    assert dataset_name in velocities
    assert "global - Average" in velocities[dataset_name]
    assert velocities[dataset_name]["global - Average"].shape == (1, 3, 3)
    assert "angular x6 - Average" in velocities[dataset_name]
    assert velocities[dataset_name]["angular x6 - Average"].shape == (1, 3, 3)
    # assert "radial x4 - Average" in velocities[dataset_name]
    # assert velocities[dataset_name]["radial x4 - Average"].shape == (1, 3, 3)


def test_mean_velocities():
    from numpy import array, sum
    from numpy.random import randint, random
    from strainmap.models.contour_mask import cylindrical_projection
    from strainmap.models.velocities import mean_velocities

    N = 3
    cartvel = random((3, 5, 512, 512))
    labels = randint(0, N, (5, 512, 512))

    origin = array((256, 256))

    velocities = mean_velocities(
        cartvel, labels, origin=origin, component_axis=0, image_axes=(2, 3)
    )

    assert velocities.shape == (len(set(labels.flat)), 3, 5)

    nribbon = sum(labels > 0, axis=(1, 2))
    bulk = sum(cartvel * (labels > 0)[None, :, :, :], axis=(2, 3)) / nribbon
    cylvel = cylindrical_projection(
        cartvel - bulk[:, :, None, None],
        origin=origin,
        component_axis=0,
        image_axes=(2, 3),
    )
    global_mean = (cylvel * (labels > 0)[None, :, :, :]).sum(axis=(2, 3)) / nribbon
    assert velocities[0, :, :] == approx(global_mean)

    nmeans = (labels == 1).sum(axis=(1, 2))
    region1_mean = (cylvel * (labels == 1)[None, :, :, :]).sum(axis=(2, 3)) / nmeans
    assert velocities[1, :, :] == approx(region1_mean)


def test_marker():
    from strainmap.models.velocities import marker
    import numpy as np

    vel = np.sin(np.linspace(0, 2 * np.pi, 50))

    idx = np.argmin(abs(vel - np.sin(np.pi / 2)))
    expected = (idx, vel[idx], 0)
    assert marker(vel, low=1, high=20, maximum=True) == expected

    idx = np.argmin(abs(vel - np.sin(3 * np.pi / 2)))
    expected = (idx, vel[idx], 0)
    assert marker(vel, low=30, high=45, maximum=False) == expected


def test_marker_es():
    from strainmap.models.velocities import marker_es
    import numpy as np

    vel = 10 * np.sin(np.linspace(0, 4 * np.pi, 50))

    pd_idx = np.argmin(abs(vel[1:25] - 10 * np.sin(3 * np.pi / 2)))
    pd = (pd_idx, vel[pd_idx], 0)
    idx = np.argmin(abs(vel[1:25] - 10 * np.sin(np.pi))) + 1
    expected = (idx, vel[idx], 0)
    assert marker_es(vel, pd) == expected


def test_marker_pc3():
    from strainmap.models.velocities import marker_pc3
    import numpy as np

    vel = 10 * np.sin(np.linspace(0, 4 * np.pi, 50))

    es_idx = np.argmin(abs(vel[1:25] - 10 * np.sin(np.pi))) + 1
    es = (es_idx, vel[es_idx], 0)
    idx = np.argmin(abs(vel[1:25] - 10 * np.sin(3 * np.pi / 2))) + 1
    expected = (idx, vel[idx], 0)
    assert marker_pc3(vel, es) == expected
    assert marker_pc3(vel / 100, es) == (np.nan, np.nan, 0)


def test_normalised_times(markers):
    from strainmap.models.velocities import normalised_times

    actual = normalised_times(markers, 50)
    assert actual[:, :, 2] == approx(markers[:, :, 2])


def test_markers_positions_internal(markers, velocity):
    from strainmap.models.velocities import _markers_positions

    actual = _markers_positions(velocity)
    assert actual[:, :, 0] == approx(markers[:, :, 0])


def test_markers_positions(markers, velocity):
    from strainmap.models.velocities import markers_positions

    actual = markers_positions(velocity[None])
    assert actual[0][:, :, 0] == approx(markers[:, :, 0])


def test_update_marker(strainmap_data, markers, velocity):
    from strainmap.models.velocities import update_marker
    import numpy as np
    from copy import deepcopy

    dataset = list(strainmap_data.data_files.keys())[0]
    strainmap_data.velocities[dataset]["global"] = [velocity]
    strainmap_data.markers[dataset]["global"] = [deepcopy(markers)]

    data = update_marker(strainmap_data, dataset, "global", 0, 0, 0, 2)
    assert data.markers[dataset]["global"][0][0, 0, 0] == 2
    assert data.markers[dataset]["global"][0][0, 0, 2] != markers[0, 0, 2]
    assert np.all(data.markers[dataset]["global"][0][0, 1:, 2] == markers[0, 1:, 2])

    data = update_marker(strainmap_data, dataset, "global", 0, 1, 3, 15)
    assert data.markers[dataset]["global"][0][1, 3, 0] == 15
    assert data.markers[dataset]["global"][0][1, 3, 2] == 350
    assert np.all(data.markers[dataset]["global"][0][0, :3, 2] != markers[0, :3, 2])

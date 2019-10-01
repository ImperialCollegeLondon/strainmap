from _pytest.python_api import approx


def test_find_theta0():
    from strainmap.models.velocities import find_theta0
    import numpy as np

    N = 5
    expected = np.random.uniform(0, 2 * np.pi, N)

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

    mask, origin = global_masks_and_origin([c1.xy], [c2.xy], shape)

    assert len(mask) == 1
    assert mask[0].shape == shape
    assert len(origin) == 1
    assert origin[0] == approx(np.array(centre), abs=0.5)


def test_transform_to_cylindrical():
    from strainmap.models.velocities import transform_to_cylindrical
    from strainmap.models.contour_mask import cylindrical_projection
    import numpy as np

    cartvel = np.random.random((3, 5, 48, 48))
    masks = np.random.randint(0, 2, (5, 48, 48))
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
        radial_regions=[4],
    ).velocities

    assert dataset_name in velocities
    assert "global" in velocities[dataset_name]
    assert velocities[dataset_name]["global"].shape == (3, 3)
    assert "angular x6" in velocities[dataset_name]
    assert velocities[dataset_name]["angular x6"].shape == (6, 3, 3)
    assert "radial x4" in velocities[dataset_name]
    assert velocities[dataset_name]["radial x4"].shape == (4, 3, 3)


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

from _pytest.python_api import approx


def test_theta_origin():
    from strainmap.models.velocities import theta_origin
    import numpy as np
    import xarray as xr

    N = 5
    expected = np.random.uniform(-np.pi, np.pi, N)

    centroid = xr.DataArray(
        np.zeros((N, 2)),
        dims=("frame", "coord"),
        coords={"coord": ["row", "col"], "frame": np.arange(N)},
    )
    septum = xr.DataArray(
        np.stack((np.cos(expected), np.sin(expected)), axis=1),
        dims=("frame", "coord"),
        coords={"coord": ["row", "col"], "frame": np.arange(N)},
    )

    assert theta_origin(centroid, septum).data == approx(expected)


def test_process_phases(strainmap_data):
    from strainmap.models.velocities import process_phases
    from strainmap.coordinates import Comp

    cine = strainmap_data.data_files.datasets[0]
    phase = process_phases(
        strainmap_data.data_files.images(cine).sel(comp=[Comp.X, Comp.Y, Comp.Z]),
        strainmap_data.sign_reversal,
    )
    assert (phase >= -0.5).all()
    assert (phase <= 0.5).all()

    phase2 = process_phases(
        strainmap_data.data_files.images(cine).sel(comp=[Comp.X, Comp.Y, Comp.Z]),
        strainmap_data.sign_reversal,
        swap=True,
    )
    assert (phase.sel(comp=Comp.X) == phase2.sel(comp=Comp.Y)).all()

    strainmap_data.sign_reversal.loc[{"comp": Comp.Z}] = -1
    phase3 = process_phases(
        strainmap_data.data_files.images(cine).sel(comp=[Comp.X, Comp.Y, Comp.Z]),
        strainmap_data.sign_reversal,
    )
    assert (phase.sel(comp=Comp.Z) == -phase3.sel(comp=Comp.Z)).all()


def test_global_mask(segmented_data):
    from strainmap.models.velocities import global_mask
    import xarray as xr
    import numpy as np

    cine = segmented_data.data_files.datasets[0]
    image = segmented_data.data_files.images(cine)
    shape = image.sizes["row"], image.sizes["col"]
    segments = segmented_data.segments.isel(cine=0)

    mask = xr.DataArray(
        np.full((len(segments.frame), shape[0], shape[1]), False, dtype=bool),
        dims=["frame", "row", "col"],
        coords={"frame": segments.frame},
    )
    global_mask(segments, shape, mask)

    # The mean of the epi and endo-cardium should be wholly within the mask
    for i in segments.frame:
        mid = segments.isel(frame=i).mean("side").astype(int)
        assert (
            mask.isel(frame=i)
            .sel(row=mid.sel(coord="row"), col=mid.sel(coord="col"))
            .all()
        )

    # The centroid should be outside of the mask
    centroid = segmented_data.septum.isel(cine=0).astype(int)
    assert ~mask.sel(row=centroid.sel(coord="row"), col=centroid.sel(coord="col")).all()


def test_find_masks(segmented_data):
    from strainmap.models.velocities import find_masks, theta_origin
    from strainmap.coordinates import Region
    import xarray as xr

    cine = segmented_data.data_files.datasets[0]
    image = segmented_data.data_files.images(cine)
    shape = image.sizes["row"], image.sizes["col"]
    segments = segmented_data.segments.isel(cine=0)
    centroid = segmented_data.centroid.isel(cine=0)
    septum = segmented_data.septum.isel(cine=0)
    theta0 = theta_origin(centroid, septum)
    mask = find_masks(segments, centroid, theta0, shape)

    global_int = mask.sel(region=Region.GLOBAL).astype(int).drop("region")
    for r in Region:
        if r == Region.GLOBAL:
            continue

        # No regional masks of one type overlap
        m = mask.sel(region=r).astype(int).sum("region")
        assert m.max() == 1

        # All regional masks add up to exactly the global mask
        xr.testing.assert_equal(m, global_int)


def test_cartesian_to_cylindrical(segmented_data, masks, theta0):
    import numpy as np
    import xarray as xr
    from strainmap.models.velocities import process_phases, cartesian_to_cylindrical
    from strainmap.coordinates import Region, Comp

    cine = segmented_data.data_files.datasets[0]
    centroid = segmented_data.centroid.isel(cine=0)
    global_mask = masks.sel(region=Region.GLOBAL).drop("region")
    phase = process_phases(
        segmented_data.data_files.images(cine).sel(comp=[Comp.X, Comp.Y, Comp.Z]),
        segmented_data.sign_reversal,
    )

    cyl = cartesian_to_cylindrical(centroid, theta0, global_mask, phase)

    # The z component should be identical
    xr.testing.assert_equal(
        xr.where(global_mask, phase, 0).sel(comp=Comp.Z).drop("comp"),
        cyl.sel(comp=Comp.LONG).drop("comp"),
    )

    # The magnitude of the in-plane components should also be identical, numerical
    # inaccuracies aside
    bulk = xr.where(global_mask, phase, np.nan).mean(dim=("row", "col"))
    bulk.loc[{"comp": Comp.Z}] = 0
    mag_cart = (phase - bulk).sel(comp=Comp.X) ** 2 + (phase - bulk).sel(
        comp=Comp.Y
    ) ** 2
    mag_cart = xr.where(global_mask, mag_cart, 0)
    mag_cyl = cyl.sel(comp=Comp.RAD) ** 2 + cyl.sel(comp=Comp.CIRC) ** 2
    xr.testing.assert_allclose(mag_cart, mag_cyl)

    return


def test_calculate_velocities(segmented_data):
    from strainmap.models.velocities import calculate_velocities

    cine = segmented_data.data_files.datasets[0]
    calculate_velocities(segmented_data, cine)


def test_initialise_markers(segmented_data):
    from strainmap.models.velocities import calculate_velocities, initialise_markers

    cine = segmented_data.data_files.datasets[0]
    calculate_velocities(segmented_data, cine)


def test_marker():
    from strainmap.models.velocities import marker_x, _MSearch
    import numpy as np

    vel = np.sin(np.linspace(0, 2 * np.pi, 50))

    idx = np.argmin(abs(vel - np.sin(np.pi / 2)))
    expected = (idx, vel[idx], 0)
    assert marker_x(vel, low=1, high=20, maximum=True) == expected

    idx = np.argmin(abs(vel - np.sin(3 * np.pi / 2)))
    expected = (idx, vel[idx], 0)
    assert marker_x(vel, low=30, high=45, maximum=False) == expected


def test_marker_es():
    from strainmap.models.velocities import marker_es, MARKERS_OPTIONS
    import numpy as np

    vel = 10 * np.sin(np.linspace(0, 4 * np.pi, 50))

    pd_idx = np.argmin(abs(vel[1:25] - 10 * np.sin(3 * np.pi / 2)))
    pd = (pd_idx, vel[pd_idx], 0)
    idx = MARKERS_OPTIONS["ES"]["low"]
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
    assert marker_pc3(vel / 100, es) == (0, np.nan, 0)


def test_normalised_times(markers):
    from strainmap.models.velocities import normalise_times

    actual = normalise_times(markers, 50)
    assert actual[:, :, 2] == approx(markers[:, :, 2])

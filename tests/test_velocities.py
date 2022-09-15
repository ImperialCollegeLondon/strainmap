from unittest.mock import patch

from pytest import approx, mark

from strainmap.coordinates import Comp, Mark, Region


def test_theta_origin():
    import numpy as np
    import xarray as xr

    from strainmap.models.transformations import theta_origin

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
    from strainmap.coordinates import Comp
    from strainmap.models.velocities import process_phases

    cine = strainmap_data.data_files.datasets[0]
    phase = process_phases(
        strainmap_data.data_files.images(cine).sel(
            comp=[Comp.X.name, Comp.Y.name, Comp.Z.name]
        ),
        strainmap_data.sign_reversal,
    )
    assert (phase >= -0.5).all()
    assert (phase <= 0.5).all()

    phase2 = process_phases(
        strainmap_data.data_files.images(cine).sel(
            comp=[Comp.X.name, Comp.Y.name, Comp.Z.name]
        ),
        strainmap_data.sign_reversal,
        swap=True,
    )
    assert (phase.sel(comp=Comp.X.name) == phase2.sel(comp=Comp.Y.name)).all()

    strainmap_data.sign_reversal.loc[{"comp": Comp.Z.name}] = -1
    phase3 = process_phases(
        strainmap_data.data_files.images(cine).sel(
            comp=[Comp.X.name, Comp.Y.name, Comp.Z.name]
        ),
        strainmap_data.sign_reversal,
    )
    assert (phase.sel(comp=Comp.Z.name) == -phase3.sel(comp=Comp.Z.name)).all()


def test_global_mask(segmented_data):
    import numpy as np
    import xarray as xr

    from strainmap.models.velocities import global_mask

    cine = segmented_data.data_files.datasets[0]
    image = segmented_data.data_files.images(cine)
    shape = image.sizes["row"], image.sizes["col"]
    segments = segmented_data.segments.isel(cine=0)

    mask = xr.DataArray(
        np.full((len(segments.frame), shape[0], shape[1]), False, dtype=bool),
        dims=["frame", "row", "col"],
        coords={"frame": segments.frame},
    )
    indices = global_mask(segments)[1:]
    mask.data[tuple(indices)] = True

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
    import xarray as xr

    from strainmap.coordinates import Region
    from strainmap.models.transformations import theta_origin
    from strainmap.models.velocities import find_masks

    segments = segmented_data.segments.isel(cine=0)
    centroid = segmented_data.centroid.isel(cine=0)
    septum = segmented_data.septum.isel(cine=0)
    theta0 = theta_origin(centroid, septum)
    mask = find_masks(segments, centroid, theta0)

    global_int = mask.sel(region=Region.GLOBAL.name).astype(int).drop("region")
    for r in Region:
        if r == Region.GLOBAL:
            continue

        # No regional masks of one type overlap
        m = mask.sel(region=r.name).astype(int).sum("region")
        assert m.max() == 1

        # All regional masks add up to exactly the global mask
        xr.testing.assert_equal(m, global_int)


def test_cartesian_to_cylindrical(segmented_data, masks):
    import numpy as np
    import xarray as xr

    from strainmap.coordinates import Comp, Region
    from strainmap.models.velocities import (
        cartesian_to_cylindrical,
        process_phases,
    )

    cine = segmented_data.data_files.datasets[0]
    centroid = segmented_data.centroid.isel(cine=0)
    global_mask = masks.sel(region=Region.GLOBAL.name).drop("region")
    phase = process_phases(
        segmented_data.data_files.images(cine).sel(
            comp=[Comp.X.name, Comp.Y.name, Comp.Z.name]
        ),
        segmented_data.sign_reversal,
    ).drop("cine")

    cyl = cartesian_to_cylindrical(centroid, global_mask, phase)
    phase_masked = xr.where(
        global_mask, phase.sel(row=global_mask.row, col=global_mask.col), 0.0
    )

    # The z component within the mask should be identical
    xr.testing.assert_equal(
        phase_masked.sel(comp=Comp.Z.name).drop("comp"),
        cyl.sel(comp=Comp.LONG.name).drop("comp"),
    )

    # The magnitude of the in-plane components should also be identical within the mask,
    # numerical inaccuracies aside
    bulk = xr.where(global_mask, phase_masked, np.nan).mean(dim=("row", "col"))
    bulk.loc[{"comp": Comp.Z.name}] = 0
    mag_cart = (phase_masked - bulk).sel(comp=Comp.X.name) ** 2 + (
        phase_masked - bulk
    ).sel(comp=Comp.Y.name) ** 2
    mag_cart = xr.where(global_mask, mag_cart, 0)
    mag_cyl = cyl.sel(comp=Comp.RAD.name) ** 2 + cyl.sel(comp=Comp.CIRC.name) ** 2
    assert mag_cart.data == approx(mag_cyl.data)


def test_marker_x(velocities):
    import numpy as np
    import xarray as xr

    from strainmap.models.velocities import _MSearch, marker_x

    comp = np.random.choice(velocities.comp.data, 2, replace=False)
    options = _MSearch(0, 25, xr.DataArray.argmax, comp)
    idx, vel = marker_x(velocities, options)

    eidx = velocities.argmax(dim="frame")[0, 0].item()
    evel = velocities.max()
    assert (idx.sel(comp=comp) == eidx).all()
    assert (vel.sel(comp=comp) == evel).all()


def test_marker_es(velocities):
    from strainmap.coordinates import Comp
    from strainmap.models.velocities import _MSearch, marker_es

    # Picks PD, so defaults to the zero crossing point
    idx, vel = marker_es(velocities, _MSearch(22, 35, None, [Comp.RAD]), 30)
    eidx = 25
    evel = velocities.sel(frame=eidx)
    assert (idx == eidx).all()
    assert (vel == evel).all()

    # Picks a local minimum
    eidx = 19
    evel = -0.3
    velocities.loc[{"frame": [eidx - 1, eidx, eidx + 1]}] = evel
    idx, vel = marker_es(velocities, _MSearch(14, 22, None, [Comp.RAD]), 30)
    assert (idx == eidx).all()
    assert (vel == evel).all()


def test_marker_pc3(empty_markers, velocities):
    from strainmap.coordinates import Mark
    from strainmap.models.velocities import MARKERS_OPTIONS, marker_pc3

    idx, vel = marker_pc3(velocities, MARKERS_OPTIONS[Mark.PC3], 30)

    eidx = velocities.argmin(dim="frame")[0, 0].item()
    evel = velocities.min()
    assert (idx == eidx).all()
    assert (vel == evel).all()


def test_initialise_markers(velocities, empty_markers):
    import numpy as np
    import xarray as xr

    from strainmap.models.velocities import initialise_markers

    markers = initialise_markers(velocities)
    assert markers.dims == empty_markers.dims
    for c in markers.coords:
        assert (markers.coords[c] == empty_markers.coords[c]).all()
    assert not xr.apply_ufunc(np.isnan, markers).all()


def test_normalised_times(empty_markers):
    from strainmap.coordinates import Mark
    from strainmap.models.velocities import normalise_times

    empty_markers.loc[{"marker": Mark.ES.name, "quantity": "frame"}] = 30
    empty_markers.loc[{"marker": Mark.PS.name, "quantity": "frame"}] = 15
    empty_markers.loc[{"marker": Mark.PD.name, "quantity": "frame"}] = 40

    normalise_times(empty_markers, 50)

    assert (empty_markers.sel(marker=Mark.ES.name, quantity="time") == 350).all()
    assert (empty_markers.sel(marker=Mark.PS.name, quantity="time") == 175).all()
    assert (empty_markers.sel(marker=Mark.PD.name, quantity="time") == 675).all()


def test_calculate_velocities(segmented_data):
    from strainmap.models.velocities import calculate_velocities

    cine = segmented_data.data_files.datasets[0]
    calculate_velocities(segmented_data, cine)

    for var in ("masks", "cylindrical", "velocities", "markers"):
        assert hasattr(getattr(segmented_data, var), "cine")


@mark.parametrize("label", (Mark.PS, Mark.ES))
@patch("strainmap.models.velocities.normalise_times", lambda m, f: m * 2)
def test_update_markers(velocities, label):
    import xarray as xr
    from numpy.random import choice, randint

    from strainmap.models.velocities import initialise_markers, update_markers

    markers = initialise_markers(velocities)
    markers2 = markers * 2

    component = Comp[choice(markers.comp)]
    region = Region[choice(markers.region)]
    location = randint(velocities.sizes["frame"])
    value = (
        velocities.sel(frame=location, region=region.name, comp=component.name)
        .isel(region=0, missing_dims="ignore")
        .item()
    )

    update_markers(
        markers=markers,
        marker_label=label,
        component=component,
        region=region,
        iregion=0,
        location=location,
        velocity_value=value,
        frames=velocities.sizes["frame"],
    )

    assert (
        markers.sel(
            region=region.name, comp=component.name, marker=label.name, quantity="frame"
        )
        == location
    ).any()
    assert (
        markers.sel(
            region=region.name,
            comp=component.name,
            marker=label.name,
            quantity="velocity",
        )
        == value
    ).any()
    if label == Mark.ES:
        xr.testing.assert_equal(
            markers2.sel(quantity="time"), markers.sel(quantity="time")
        )

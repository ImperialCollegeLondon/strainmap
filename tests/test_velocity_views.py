def test_create_velocity_plot(dicom_data_path):
    """Checks creating the velocity plot does throw."""
    from strainmap.models.readers import (
        read_dicom_directory_tree,
        read_all_images,
        images_to_numpy,
    )
    from strainmap.models.contour_mask import radial_segments, angular_segments, Contour
    from strainmap.gui.velocity_view import plot_velocities

    cartesian = images_to_numpy(
        read_all_images(read_dicom_directory_tree(dicom_data_path))
    )['ParallelSpirals_R3_MID'].phase

    outer = Contour.circle((320, 350), 100)
    inner = Contour.circle((315, 350), 60)
    rseg = radial_segments(outer, inner, center=(315, 350), shape=cartesian.shape[2:])
    tseg = angular_segments(3, (315, 350), shape=cartesian.shape[2:])
    labels = rseg + tseg * max(set(rseg.flatten()))
    labels[rseg == 0] = 0
    plot_velocities(cartesian, labels, origin=(315, 350), image_axes=(2, 3))

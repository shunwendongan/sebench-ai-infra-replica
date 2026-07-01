import numpy as np

from sebench_infra.spatial.fixtures import make_rotated_cube
from sebench_infra.spatial.geometry import GeometryBridge, centroid_delta, kabsch_rotation


def test_kabsch_recovers_known_z_rotation() -> None:
    source, target, expected_rotation = make_rotated_cube()

    estimated = kabsch_rotation(source, target)

    assert np.allclose(estimated, expected_rotation, atol=1e-8)


def test_geometry_bridge_reports_centroid_delta_and_prefix() -> None:
    source, target, _ = make_rotated_cube()

    delta = centroid_delta(source, target)
    facts = GeometryBridge().compute(source, target)

    assert np.allclose([facts.dx, facts.dy, facts.dz], delta)
    assert "Structured spatial facts" in facts.to_prefix()

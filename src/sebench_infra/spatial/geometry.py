from dataclasses import dataclass

import numpy as np


def centroid(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points.mean(axis=0)


def centroid_delta(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return centroid(target) - centroid(source)


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Estimate the rotation matrix that aligns source points to target points.

    The implementation follows the Kabsch SVD solution and corrects reflections.
    """

    source_points = np.asarray(source, dtype=float)
    target_points = np.asarray(target, dtype=float)
    if source_points.shape != target_points.shape:
        raise ValueError("source and target must have identical shape")
    if source_points.ndim != 2 or source_points.shape[1] != 3 or source_points.shape[0] < 3:
        raise ValueError("point sets must have shape (N, 3), N >= 3")

    source_centered = source_points - centroid(source_points)
    target_centered = target_points - centroid(target_points)
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    reflection_fix = np.eye(3)
    reflection_fix[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ reflection_fix @ u.T


@dataclass(frozen=True)
class GeometryFacts:
    dx: float
    dy: float
    dz: float
    rotation_matrix: list[list[float]]
    rotation_trace: float

    def to_prefix(self) -> str:
        return (
            "Structured spatial facts: "
            f"dx={self.dx:.6f}, dy={self.dy:.6f}, dz={self.dz:.6f}, "
            f"rotation_trace={self.rotation_trace:.6f}."
        )


class GeometryBridge:
    """Convert low-level 3D point changes into LLM-readable structured facts."""

    def compute(self, source: np.ndarray, target: np.ndarray) -> GeometryFacts:
        delta = centroid_delta(source, target)
        rotation = kabsch_rotation(source, target)
        return GeometryFacts(
            dx=float(delta[0]),
            dy=float(delta[1]),
            dz=float(delta[2]),
            rotation_matrix=np.round(rotation, 8).tolist(),
            rotation_trace=float(np.trace(rotation)),
        )

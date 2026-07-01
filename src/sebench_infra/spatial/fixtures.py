import numpy as np


def make_rotated_cube(
    theta_degrees: float = 90.0,
    translation: tuple[float, float, float] = (1, 2, 3),
):
    theta = np.deg2rad(theta_degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    target = source @ rotation.T + np.asarray(translation)
    return source, target, rotation

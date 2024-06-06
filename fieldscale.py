"""
Source code for the paper:

Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image
Hyeonjae Gil, Myeon-Hwan Jeon, and Ayoung Kim

Please cite the paper if you use this code.

@article{gil2024fieldscale,
  title={Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image},
  author={Gil, Hyeonjae and Jeon, Myung-Hwan and Kim, Ayoung},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}

Original author: Hyeonjae Gil
Author email: h.gil@snu.ac.kr
"""

import numpy as np
import cv2
from typing import Literal, Union


def gridwise_min(image: np.ndarray, grid_shape: tuple = (1, 1)) -> np.ndarray:
    """Return the minimum value of each patch in the image."""
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output = np.zeros(grid_shape, dtype=image.dtype)
    for i, j in np.ndindex(grid_shape):
        output[i, j] = np.amin(image[
            patch_shape[0] * i : patch_shape[0] * (i + 1),
            patch_shape[1] * j : patch_shape[1] * (j + 1)
        ])
    return output


def gridwise_max(image: np.ndarray, grid_shape: tuple = (1, 1)) -> np.ndarray:
    """Return the maximum value of each patch in the image."""
    patch_shape = (image.shape[0] // grid_shape[0], image.shape[1] // grid_shape[1])
    output = np.zeros(grid_shape, dtype=image.dtype)
    for i, j in np.ndindex(grid_shape):
        output[i, j] = np.amax(image[
            patch_shape[0] * i : patch_shape[0] * (i + 1),
            patch_shape[1] * j : patch_shape[1] * (j + 1)
        ])
    return output


def get_neighbor_grids(grid: np.ndarray, xy: tuple, max_distance: int = 1) -> list:
    """Return the neighbors of a pixel in a grid."""
    h, w = grid.shape
    x, y = xy
    neighbors = [
        (x + i, y + j)
        for i in range(-max_distance, max_distance + 1)
        for j in range(-max_distance, max_distance + 1)
        if (i != 0 or j != 0) and (0 <= x + i < h) and (0 <= y + j < w)
    ]
    return sorted(neighbors, key=lambda k: (k[0], k[1]))


def local_extrema_suppression(grid: np.ndarray, 
                              local_distance: int,
                              diff_threshold: float, 
                              extrema: Literal['max', 'min']) -> np.ndarray:
    """Clip the extreme values in the grid."""
    assert extrema in ['max', 'min']
    if local_distance <= 0 or diff_threshold <= 0:
        return grid

    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=local_distance)
        neighbor_values = np.array([grid[xy] for xy in neighbors])
        if extrema == 'max' and grid[i, j] >= neighbor_values.max():
            diff = grid[i, j] - neighbor_values.mean()
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() + diff_threshold
        elif extrema == 'min' and grid[i, j] <= neighbor_values.min():
            diff = neighbor_values.mean() - grid[i, j]
            if diff > diff_threshold:
                grid[i, j] = neighbor_values.mean() - diff_threshold
    return grid


def message_passing(grid: np.ndarray, 
                    direction: Literal['increase', 'decrease']) -> np.ndarray:
    """Message passing algorithm for grid."""
    assert direction in ['increase', 'decrease']
    grid_new = np.zeros_like(grid, dtype=np.float64)

    for i, j in np.ndindex(grid.shape):
        neighbors = get_neighbor_grids(grid, (i, j), max_distance=1)
        neighbors_value = [grid[neighbor] for neighbor in neighbors]
        mean = np.mean(neighbors_value + [grid[i, j]])
        bigger, smaller = (mean, grid[i, j]) if mean > grid[i, j] else (grid[i, j], mean)
        grid_new[i, j] = bigger if direction == 'increase' else smaller

    return grid_new


def rescale_image_with_fields(image: np.ndarray, 
                              min_field: np.ndarray, 
                              max_field: np.ndarray) -> np.ndarray:
    """Rescale the image with min_field and max_field as the lower and upper bound."""
    assert image.shape == min_field.shape == max_field.shape

    image = image.astype(np.float64)
    min_field = min_field.astype(np.float64)
    max_field = max_field.astype(np.float64)

    min_field = np.where(min_field > max_field, max_field, min_field)
    max_field = np.where(max_field < min_field, min_field, max_field)
    image = np.clip(image, min_field, max_field)
    image = (image - min_field) / (max_field - min_field) * 255

    return image.astype(np.uint8)


class Fieldscale:
    def __init__(self, max_diff: float = 400, min_diff: float = 400,
                 iteration: int = 7, gamma: float = 1.5,
                 clahe: bool = True, video: bool = False):
        assert max_diff >= 0 and isinstance(max_diff, (int, float))
        assert min_diff >= 0 and isinstance(min_diff, (int, float))
        assert iteration > 0 and isinstance(iteration, int)
        assert gamma > 0 and isinstance(gamma, (int, float))
        assert isinstance(clahe, bool)
        assert isinstance(video, bool)

        self.max_diff = max_diff
        self.min_diff = min_diff
        self.iteration = iteration
        self.gamma = gamma
        self.clahe = clahe
        self.video = video
        self.prev_min_field = None
        self.prev_max_field = None


    def __call__(self, input: Union[str, np.ndarray]) -> np.ndarray:
        """Process an image or a path to an image."""
        if isinstance(input, str):
            image = cv2.imread(input, -1)
            if image is None:
                raise ValueError(f"Unable to read image from path: {input}")
        elif isinstance(input, np.ndarray):
            image = input
        else:
            raise TypeError("Input should be a file path or an numpy.ndarray.")

        min_grid = gridwise_min(image, (8, 8))
        max_grid = gridwise_max(image, (8, 8))

        max_grid = local_extrema_suppression(
            max_grid, local_distance=2, diff_threshold=self.max_diff, extrema='max'
        )
        max_grid = local_extrema_suppression(
            max_grid, local_distance=2, diff_threshold=self.min_diff, extrema='min'
        )

        for _ in range(self.iteration):
            min_grid = message_passing(min_grid, direction='decrease').astype(np.float64)
            max_grid = message_passing(max_grid, direction='increase').astype(np.float64)

        min_field = cv2.resize(min_grid, dsize=(image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        max_field = cv2.resize(max_grid, dsize=(image.shape[1], image.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        if self.video and self.prev_min_field is not None:
            min_field = 0.1 * min_field + 0.9 * self.prev_min_field
            max_field = 0.1 * max_field + 0.9 * self.prev_max_field

        self.prev_min_field = min_field
        self.prev_max_field = max_field

        rescaled = rescale_image_with_fields(image, min_field, max_field)

        if self.gamma > 0:
            rescaled = (255 * np.power(rescaled.astype(np.float64) / 255, self.gamma)).astype(np.uint8)

        if self.clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            rescaled = clahe.apply(rescaled)

        return rescaled
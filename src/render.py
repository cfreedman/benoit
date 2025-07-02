import time
from typing import Callable, Literal

import numpy as np
from numba import jit, cuda

from src.ascii import get_ascii
from src.grid import GridPoints, generate_grid
from src.mandelbrot import EscapeFractal


class Renderer:
    def __init__(
        self,
        fractal: EscapeFractal,
        grid: GridPoints,
        mode: Literal["normal", "jit", "gpu"],
    ):
        self.grid = grid
        self.fractal = fractal
        self.mode = mode

        self.sample_function = self.generate_sample_grid_function()
        self.render_function = self.generate_render_function()

    def generate_sample_grid_function(self):
        x_unit = self.grid.x_matrix[1, 0] - self.grid.x_matrix[0, 0]
        y_unit = self.grid.y_matrix[0, 1] - self.grid.y_matrix[0, 0]

        def generate_sample(x_center: float, y_center: float) -> np.ndarray:
            return generate_grid(
                x_center=x_center,
                y_center=y_center,
                x_length=x_unit,
                y_length=y_unit,
                x_divisions=5,
                y_divisions=5,
            )

        if self.mode == "normal":
            return generate_sample
        if self.mode == "jit":
            return jit(generate_sample)
        return cuda.jit(generate_sample)

    def generate_render_function(self):
        average_escape_function = self.fractal.generate_average_escape_function(
            self.mode
        )
        sample_grid_function = self.sample_function

        def render(x_matrix: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
            x_grid_length, y_grid_length = x_matrix.shape

            results = np.empty(shape=(y_grid_length, x_grid_length))

            for j in range(y_grid_length):
                for i in range(x_grid_length):
                    x_center = x_matrix[i, j]
                    y_center = y_matrix[i, -(j + 1)]
                    x_sample_grid, y_sample_grid, x_sample_matrix, y_sample_matrix = (
                        sample_grid_function(x_center=x_center, y_center=y_center)
                    )
                    print(x_sample_grid)
                    print(y_sample_grid)

                    results[j, i] = average_escape_function(
                        x_sample_matrix, y_sample_matrix
                    )

            return results

        if self.mode == "normal":
            return render
        if self.mode == "jit":
            return jit(render)

        return cuda.jit(render)

    def render(self):
        x_matrix = self.grid.x_matrix
        y_matrix = self.grid.y_matrix

        max_iterations = self.fractal.max_iterations

        results = self.render_function(x_matrix, y_matrix)
        print(results[0, 0])

        height, width = results.shape

        for j in range(height):
            line = "".join(
                [get_ascii(result, max_iterations) for result in results[j, :]]
            )
            print(line)


def select_variant_function(
    base_function: Callable,
    jit_function: Callable,
    gpu_function: Callable,
    mode: Literal["normal", "jit", "gpu"],
) -> Callable:
    if mode == "normal":
        return base_function
    if mode == "jit":
        return jit_function
    return gpu_function


def render(
    average_escape_function: Callable[[np.ndarray, np.ndarray], float],
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
) -> np.ndarray:
    x_grid_length, y_grid_length = x_matrix.shape
    x_unit = x_matrix[1, 0] - x_matrix[0, 0]
    y_unit = y_matrix[0, 1] - y_matrix[0, 0]

    results = np.empty(shape=(y_grid_length, x_grid_length))

    for j in range(y_grid_length):
        for i in range(x_grid_length):
            x_center = x_matrix[i, j]
            y_center = y_matrix[i, -(j + 1)]
            x_sample_grid, y_sample_grid, x_sample_matrix, y_sample_matrix = (
                generate_grid(
                    x_center=x_center,
                    y_center=y_center,
                    x_length=x_unit,
                    y_length=y_unit,
                    x_divisions=5,
                    y_divisions=5,
                )
            )
            results[j, i] = average_escape_function(x_sample_matrix, y_sample_matrix)

    return results


@jit
def render_jit(
    average_escape_function: Callable[[np.ndarray, np.ndarray], float],
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
) -> np.ndarray:
    x_length, y_length = x_matrix.shape

    results = np.empty(shape=(y_length, x_length))

    for j in range(y_length):
        for i in range(x_length):
            results[j, i] = average_escape_function(
                x_matrix[i, j], y_matrix[i, -(j + 1)]
            )

    return results


@cuda.jit
def render_gpu(
    average_escape_function: Callable[[np.ndarray, np.ndarray], float],
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
) -> np.ndarray:
    x_length, y_length = x_matrix.shape

    results = np.empty(shape=(y_length, x_length))

    for j in range(y_length):
        for i in range(x_length):
            results[j, i] = average_escape_function(
                x_matrix[i, j], y_matrix[i, -(j + 1)]
            )

    return results

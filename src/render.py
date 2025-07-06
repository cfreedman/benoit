from typing import Literal

import numpy as np

from src.ascii import get_ascii
from src.grid import GridPoints
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

        self.average_escape_function = fractal.build_average_escape_function(
            mode=self.mode
        )

    def render(self, sample_grid_function, average_escape_function):
        x_grid, y_grid = self.grid.x_grid, self.grid.y_grid
        x_unit_length, y_unit_length = self.grid.steps()

        results = np.empty(shape=(len(y_grid), len(x_grid)))

        for j in range(len(y_grid)):
            for i in range(len(x_grid)):
                x_center = x_grid[i]
                y_center = y_grid[-(j + 1)]
                x_sample_grid, y_sample_grid = sample_grid_function(
                    x_center=x_center,
                    y_center=y_center,
                    x_length=x_unit_length,
                    y_length=y_unit_length,
                )

                results[j, i] = average_escape_function(x_sample_grid, y_sample_grid)

        for j in range(len(y_grid)):
            line = "".join(
                [
                    get_ascii(value=value, max_iterations=self.fractal.max_iterations)
                    for value in results[j, :]
                ]
            )
            print(line)


def render(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    sample_grid_function,
    average_escape_function,
) -> np.ndarray:
    x_grid, y_grid = x_grid, y_grid
    x_unit_length, y_unit_length = x_grid[1] - x_grid[0], y_grid[1] - y_grid[0]

    results = np.empty(shape=(len(y_grid), len(x_grid)))

    for j in range(len(y_grid)):
        for i in range(len(x_grid)):
            x_center = x_grid[i]
            y_center = y_grid[-(j + 1)]
            x_sample_grid, y_sample_grid = sample_grid_function(
                x_center=x_center,
                y_center=y_center,
                x_length=x_unit_length,
                y_length=y_unit_length,
            )

            results[j, i] = average_escape_function(x_sample_grid, y_sample_grid)

    return results

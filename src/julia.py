from math import sqrt

import numpy as np

from src.grid import ComplexPoint, GridPoints


def calculate_boundary(c: ComplexPoint) -> float:
    """
    Solving for the valid Julia set radius satisfying R^2 - R - |c| >= 0
    """

    determinant = 1 + 4 * (c.x**2 + c.y**2)

    return (1 + sqrt(determinant)) / 2


def escape(z: ComplexPoint, c: ComplexPoint, max_iterations: int) -> int:
    boundary = calculate_boundary(c)

    z_loop = z
    iter = 0

    while z.length_squared() <= boundary**2 and iter <= max_iterations:
        z_loop = z_loop * z_loop + c
        iter += 1

    return iter


def average_escape_julia(grid: GridPoints, c: ComplexPoint, max_iterations: int) -> int:
    xv, yv = np.meshgrid(grid.x_grid, grid.y_grid, indexing="ij")

    total = 0
    for i in range(len(grid.x_grid)):
        for j in range(len(grid.y_grid)):
            z = ComplexPoint(xv[i, j], yv[i, j])
            total += escape(z, c, max_iterations)

    return total / (len(grid.x_grid) * len(grid.y_grid))

import numpy as np
from src.grid import CharacterBlock, ComplexPoint, GridPoints


def escape(c_point: ComplexPoint, max_iterations: int) -> int:
    loop_point = ComplexPoint(0, 0)
    iter = 0

    while loop_point.length_squared() <= 4 and iter < max_iterations:
        loop_point = loop_point * loop_point + c_point
        iter += 1

    return iter


def average_escape(grid: GridPoints, max_iterations: int) -> int:
    xv, yv = np.meshgrid(grid.x_grid, grid.y_grid, indexing="ij")

    total = 0
    for i in range(len(grid.x_grid)):
        for j in range(len(grid.y_grid)):
            c_point = ComplexPoint(xv[i, j], yv[i, j])
            total += escape(c_point, max_iterations)

    return total / (len(grid.x_grid) * len(grid.y_grid))

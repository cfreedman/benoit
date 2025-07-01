from typing import Callable
from src.grid import ComplexPoint, GridHalfSteps, GridPoints, generate_sample_grid


class EscapeFractal:
    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def escape(self, point: ComplexPoint) -> int:
        pass

    def average_escape(self, grid: GridPoints) -> int:
        x_matrix, y_matrix = grid.x_matrix, grid.y_matrix

        total = 0

        for i in range(len(grid.x_grid)):
            for j in range(len(grid.y_grid)):
                point = ComplexPoint(x_matrix[i, j], y_matrix[i, j])
                total += self.escape(point)

        return total / (len(grid.x_grid) * len(grid.y_grid))


def compute_escape(
    center: ComplexPoint, half_steps: GridHalfSteps, fractal: EscapeFractal
) -> int:
    grid = generate_sample_grid(center, half_steps)

    return fractal.average_escape(grid)


class Mandelbrot(EscapeFractal):
    def __init__(self, max_iterations: int):
        super().__init__(max_iterations)

    def escape(self, point: ComplexPoint) -> int:
        loop_point = ComplexPoint(0, 0)

        iter = 0
        while loop_point.length_squared() <= 4 and iter < self.max_iterations:
            loop_point = loop_point * loop_point + point
            iter += 1

        return iter

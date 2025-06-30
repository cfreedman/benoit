from math import sqrt

import numpy as np

from src.grid import ComplexPoint, GridPoints
from src.mandelbrot import EscapeFractal


class JuliaSet(EscapeFractal):
    def __init__(self, max_iterations: int, parameter_point: ComplexPoint):
        super().__init__(max_iterations)
        self.parameter_point = parameter_point

    @staticmethod
    def calculate_boundary(c: ComplexPoint) -> float:
        """
        Solving for the valid Julia set radius satisfying R^2 - R - |c| >= 0
        """

        determinant = 1 + 4 * (c.x**2 + c.y**2)

        return (1 + sqrt(determinant)) / 2

    def escape(self, point: ComplexPoint) -> int:
        boundary = self.calculate_boundary(self.parameter_point)

        z_loop = point
        iter = 0

        while z_loop.length_squared() <= boundary**2 and iter <= self.max_iterations:
            z_loop = z_loop * z_loop + self.parameter_point
            iter += 1

        return iter

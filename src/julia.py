from math import sqrt

import numpy as np
from numba import jit, cuda

from src.grid import ComplexPoint, GridPoints
from src.mandelbrot import EscapeFractal


# class JuliaSet(EscapeFractal):
#     # @staticmethod
#     # def calculate_boundary(c: ComplexPoint) -> float:
#     #     """
#     #     Solving for the valid Julia set radius satisfying R^2 - R - |c| >= 0
#     #     """

#     #     determinant = 1 + 4 * (c.x**2 + c.y**2)

#     #     return (1 + sqrt(determinant)) / 2

#     # def escape(self, point: ComplexPoint) -> int:
#     #     boundary = self.calculate_boundary(self.parameter_point)

#     #     z_loop = point
#     #     iter = 0

#     #     while z_loop.length_squared() <= boundary**2 and iter < self.max_iterations:
#     #         z_loop = z_loop * z_loop + self.parameter_point
#     #         iter += 1

#     #     return iter

#     def escape_function_factory(self, mode, **parameters):
#         if mode == "normal":
#             def compute_escape(input_x: float, input_y: float) -> int:
#                 return julia_escape(input_x, input_y, )
#         elif mode == "jit":
#         else:


def julia_escape(
    input_x: float,
    input_y: float,
    max_iterations: int,
    parameter_x: float,
    parameter_y: float,
) -> int:
    determinant = 1 + 4 * (parameter_x**2 + parameter_y**2)
    boundary = (1 + sqrt(determinant)) / 2

    x_loop, y_loop = input_x, input_y
    iter = 0

    while (x_loop**2 + y_loop**2) <= boundary**2 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y

        iter += 1

    return iter


@jit
def julia_escape_jit(
    input_x: float,
    input_y: float,
    max_iterations: int,
    parameter_x: float,
    parameter_y: float,
) -> int:
    determinant = 1 + 4 * (parameter_x**2 + parameter_y**2)
    boundary = (1 + sqrt(determinant)) / 2

    x_loop, y_loop = input_x, input_y
    iter = 0

    while (x_loop**2 + y_loop**2) <= boundary**2 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y

        iter += 1

    return iter


@cuda.jit
def julia_escape_gpu(
    input_x: float,
    input_y: float,
    max_iterations: int,
    parameter_x: float,
    parameter_y: float,
) -> int:
    determinant = 1 + 4 * (parameter_x**2 + parameter_y**2)
    boundary = (1 + sqrt(determinant)) / 2

    x_loop, y_loop = input_x, input_y
    iter = 0

    while (x_loop**2 + y_loop**2) <= boundary**2 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y

        iter += 1

    return iter


class JuliaSet(EscapeFractal):
    def __init__(self, max_iterations: int, parameter_x: float, parameter_y: float):
        super().__init__(
            base_function=julia_escape,
            jit_function=julia_escape_jit,
            gpu_function=julia_escape_gpu,
            max_iterations=max_iterations,
            parameter_x=parameter_x,
            parameter_y=parameter_y,
        )

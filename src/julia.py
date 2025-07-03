from math import sqrt

import numpy as np
from numba import jit, cuda

from src.grid import ComplexPoint
from src.mandelbrot import EscapeFractal


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
    def __init__(self, max_iterations: int, parameter: ComplexPoint):
        super().__init__(max_iterations=max_iterations, parameter=parameter)

    def build_escape_function(self):
        def escape_function(input_x: float, input_y: float) -> int:
            return julia_escape(
                input_x,
                input_y,
                self.max_iterations,
                self.parameter.x,
                self.parameter.y,
            )

        return escape_function


class JuliaSetJIT(EscapeFractal):
    def __init__(self, max_iterations: int, parameter: ComplexPoint):
        super().__init__(max_iterations=max_iterations, parameter=parameter)

    def build_escape_function(self):
        @jit
        def escape_function(input_x: float, input_y: float) -> int:
            return julia_escape_jit(
                input_x,
                input_y,
                self.max_iterations,
                self.parameter.x,
                self.parameter.y,
            )

        return escape_function


class JuliaSetGPU(EscapeFractal):
    def __init__(self, max_iterations: int, parameter: ComplexPoint):
        super().__init__(max_iterations=max_iterations, parameter=parameter)

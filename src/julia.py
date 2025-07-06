from math import sqrt
from typing import Callable, Literal

from numba import njit, cuda

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
        x_new = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y
        x_loop = x_new

        iter += 1

    return iter


@njit
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
        x_new = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y
        x_loop = x_new

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
        x_new = x_loop**2 - y_loop**2 + parameter_x
        y_loop = 2 * x_loop * y_loop + parameter_y
        x_loop = x_new

        iter += 1

    return iter


class JuliaSet(EscapeFractal):
    def __init__(self, max_iterations: int, parameter: ComplexPoint):
        super().__init__(
            max_iterations=max_iterations,
            base_function=julia_escape,
            jit_function=julia_escape_jit,
        )
        self.parameter_x = parameter.x
        self.parameter_y = parameter.y

    def _make_escape_function(self, mode: Literal["normal", "jit", "gpu"]) -> Callable:
        max_iterations = self.max_iterations
        parameter_x, parameter_y = self.parameter_x, self.parameter_y

        if mode == "normal":

            def bound_escape_function(input_x: float, input_y: float) -> int:
                return self.base_function(
                    input_x, input_y, max_iterations, parameter_x, parameter_y
                )
        else:
            escape_function = self.jit_function

            @njit
            def bound_escape_function(input_x: float, input_y: float) -> int:
                return escape_function(
                    input_x, input_y, max_iterations, parameter_x, parameter_y
                )

        return bound_escape_function

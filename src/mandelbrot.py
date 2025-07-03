from typing import Callable, Literal

import numpy as np
from numba import njit
from numba import cuda

from src.escape_fractal import EscapeFractal


def mandelbrot_escape(input_x: float, input_y: float, max_iterations: int) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) <= 4 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * x_loop * y_loop + input_y

        iter += 1

    return iter


@njit
def mandelbrot_escape_jit(input_x: float, input_y: float, max_iterations: int) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) <= 4 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * x_loop * y_loop + input_y

        iter += 1

    return iter


@cuda.jit
def mandelbrot_escape_gpu_jit(
    input_x: float, input_y: float, max_iterations: int
) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) <= 4 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * x_loop * y_loop + input_y

        iter += 1

    return iter


class Mandelbrot(EscapeFractal):
    def __init__(self, max_iterations: int):
        super().__init__(
            max_iterations=max_iterations,
            base_function=mandelbrot_escape,
            jit_function=mandelbrot_escape_jit,
        )


class MandelbrotJIT(EscapeFractal):
    def __init__(self, max_iterations: int):
        super().__init__(
            max_iterations=max_iterations,
        )

    @staticmethod
    def build_escape_function(max_iterations: int) -> Callable:
        @njit
        def escape_function(input_x: float, input_y: float) -> int:
            return mandelbrot_escape_jit(input_x, input_y, max_iterations)

        return escape_function


class MandelbrotGPU(EscapeFractal):
    def __init__(self, max_iterations: int):
        super().__init__(
            max_iterations=max_iterations,
        )

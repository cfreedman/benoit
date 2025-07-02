from typing import Callable, Literal

import numpy as np
from numba import jit
from numba import cuda


class EscapeFractal:
    def __init__(
        self,
        base_function,
        jit_function,
        gpu_function,
        max_iterations: int,
        **parameters,
    ):
        self.base_function = base_function
        self.jit_function = jit_function
        self.gpu_function = gpu_function
        self.max_iterations = max_iterations
        self.parameters = parameters

    def generate_escape_function(
        self, mode: Literal["normal", "jit", "gpu"]
    ) -> Callable:
        def compute_escape(input_x: float, input_y: float) -> int:
            return self.base_function(
                input_x, input_y, self.max_iterations, **self.parameters
            )

        @jit
        def compute_escape_jit(input_x: float, input_y: float) -> int:
            return self.jit_function(
                input_x, input_y, self.max_iterations, **self.parameters
            )

        @cuda.jit
        def compute_escape_gpu(input_x: float, input_y: float) -> int:
            return self.gpu_function(
                input_x, input_y, self.max_iterations, **self.parameters
            )

        return (
            compute_escape
            if mode == "normal"
            else compute_escape_jit
            if mode == "jit"
            else compute_escape_gpu
        )

    def generate_average_escape_function(
        self, mode: Literal["normal", "jit", "gpu"]
    ) -> Callable:
        computed_escape_function = self.generate_escape_function(mode)

        def average_escape(
            x_matrix: np.ndarray,
            y_matrix: np.ndarray,
        ) -> float:
            x_length, y_length = x_matrix.shape

            total = 0
            for i in range(x_length):
                for j in range(y_length):
                    total += computed_escape_function(x_matrix[i, j], y_matrix[i, j])
            return total / (x_length * y_length)

        if mode == "normal":
            return average_escape
        elif mode == "jit":
            return jit(average_escape)
        else:
            return cuda.jit(average_escape)


def mandelbrot_escape(input_x: float, input_y: float, max_iterations: int) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) <= 4 and iter < max_iterations:
        x_loop = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * x_loop * y_loop + input_y

        iter += 1

    return iter


@jit
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
            base_function=mandelbrot_escape,
            jit_function=mandelbrot_escape_jit,
            gpu_function=mandelbrot_escape_gpu_jit,
            max_iterations=max_iterations,
        )

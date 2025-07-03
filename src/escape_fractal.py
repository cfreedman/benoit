from typing import Callable
from typing_extensions import Literal

import numpy as np


class EscapeFractal:
    def __init__(
        self, max_iterations: int, base_function: Callable, jit_function: Callable
    ):
        self.max_iterations = max_iterations
        self.base_function = base_function
        self.jit_function = jit_function

    def build_escape_function(
        self, max_iterations: int, mode: Literal["normal", "jit", "gpu"]
    ) -> Callable:
        if mode == "normal":

            def bound_escape_function(input_x: float, input_y: float) -> int:
                return self.base_function(input_x, input_y, max_iterations)
        else:

            def bound_escape_function(input_x: float, input_y: float) -> int:
                return self.jit_function(input_x, input_y, max_iterations)

    @staticmethod
    def average_escape_function(escape_function: Callable) -> Callable:
        def average_escape(
            x_grid: np.ndarray,
            y_grid: np.ndarray,
        ) -> float:
            x_length, y_length = len(x_grid), len(y_grid)

            total = 0
            for i in range(x_length):
                for j in range(y_length):
                    total += escape_function(x_grid[i], y_grid[j])
            return total / (x_length * y_length)

        return average_escape

    @staticmethod
    def build_average_escape_function(
        mode=Literal["normal", "jit", "gpu"],
    ) -> Callable:
        def average_escape(
            x_grid: np.ndarray,
            y_grid: np.ndarray,
        ) -> float:
            x_length, y_length = len(x_grid), len(y_grid)

            total = 0
            for i in range(x_length):
                for j in range(y_length):
                    total += base_function(x_grid[i], y_grid[j], max_iterations)
            return total / (x_length * y_length)

        return average_escape

    # def generate_escape_function(
    #     self, mode: Literal["normal", "jit", "gpu"]
    # ) -> Callable:
    #     def compute_escape(input_x: float, input_y: float) -> int:
    #         return self.base_function(
    #             input_x, input_y, self.max_iterations, **self.parameters
    #         )

    #     @jit
    #     def compute_escape_jit(input_x: float, input_y: float) -> int:
    #         return self.jit_function(
    #             input_x, input_y, self.max_iterations, **self.parameters
    #         )

    #     @cuda.jit
    #     def compute_escape_gpu(input_x: float, input_y: float) -> int:
    #         return self.gpu_function(
    #             input_x, input_y, self.max_iterations, **self.parameters
    #         )

    #     return (
    #         compute_escape
    #         if mode == "normal"
    #         else compute_escape_jit
    #         if mode == "jit"
    #         else compute_escape_gpu
    #     )

    # def generate_average_escape_function(
    #     self, mode: Literal["normal", "jit", "gpu"]
    # ) -> Callable:
    #     computed_escape_function = self.generate_escape_function(mode)

    #     def average_escape(
    #         x_matrix: np.ndarray,
    #         y_matrix: np.ndarray,
    #     ) -> float:
    #         x_length, y_length = x_matrix.shape

    #         total = 0
    #         for i in range(x_length):
    #             for j in range(y_length):
    #                 total += computed_escape_function(x_matrix[i, j], y_matrix[i, j])
    #         return total / (x_length * y_length)

    #     if mode == "normal":
    #         return average_escape
    #     elif mode == "jit":
    #         return jit(average_escape)
    #     else:
    #         return cuda.jit(average_escape)

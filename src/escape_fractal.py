from typing import Callable, Literal

import numpy as np
from numba import njit


class EscapeFractal:
    def __init__(
        self, max_iterations: int, base_function: Callable, jit_function: Callable
    ):
        self.max_iterations = max_iterations
        self.base_function = base_function
        self.jit_function = jit_function

    def build_escape_function(self, mode: Literal["normal", "jit", "gpu"]) -> Callable:
        return self._make_escape_function(mode)

    def _make_escape_function(self, mode: Literal["normal", "jit", "gpu"]) -> Callable:
        pass

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

    def build_average_escape_function(
        self,
        mode=Literal["normal", "jit", "gpu"],
    ) -> Callable:
        if mode == "normal":
            escape_function = self.build_escape_function(mode="normal")
            return self.average_escape_function(escape_function)

        escape_function = self.build_escape_function(mode="jit")
        return njit(self.average_escape_function(escape_function))

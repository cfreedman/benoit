from typing import Callable, Literal
from src.escape_fractal import EscapeFractal

from numba import njit


def burning_ship_escape(input_x: float, input_y: float, max_iterations: int) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) < 4 and iter < max_iterations:
        x_new = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * abs(x_loop) * abs(y_loop) + input_y
        x_loop = x_new

        iter += 1

    return iter


@njit
def burning_ship_escape_jit(input_x: float, input_y: float, max_iterations: int) -> int:
    x_loop = y_loop = 0

    iter = 0
    while (x_loop**2 + y_loop**2) < 4 and iter < max_iterations:
        x_new = x_loop**2 - y_loop**2 + input_x
        y_loop = 2 * abs(x_loop) * abs(y_loop) + input_y
        x_loop = x_new

        iter += 1

    return iter


class BurningShip(EscapeFractal):
    def __init__(self, max_iterations: int):
        super().__init__(
            max_iterations=max_iterations,
            base_function=burning_ship_escape,
            jit_function=burning_ship_escape_jit,
        )

    def _make_escape_function(self, mode: Literal["normal", "jit", "gpu"]) -> Callable:
        max_iterations = self.max_iterations

        if mode == "normal":

            def bound_escape_function(input_x: float, input_y: float) -> int:
                return self.base_function(input_x, input_y, max_iterations)
        else:
            escape_function = self.jit_function

            @njit
            def bound_escape_function(input_x: float, input_y: float) -> int:
                return escape_function(input_x, input_y, max_iterations)

        return bound_escape_function

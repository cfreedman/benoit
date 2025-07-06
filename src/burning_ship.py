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

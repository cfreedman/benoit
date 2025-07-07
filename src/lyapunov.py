from numba import njit
import numpy as np

from src.escape_fractal import EscapeFractal


def lyapunov_escape(
    input_x: float, input_y: float, max_iterations: int, sequence: np.ndarray
) -> float:
    sequence_length = len(sequence)

    x_loop = 0.5
    total = 0

    for i in range(max_iterations):
        r_loop = input_x if (i % sequence_length) % 2 == 0 else input_y
        if i > 0:
            total += np.log2(abs(r_loop * (1 - 2 * x_loop)))
        x_loop = r_loop * x_loop * (1 - x_loop)

    return total


@njit
def lyapunov_escape_jit(
    input_x: float, input_y: float, max_iterations: int, sequence: np.ndarray
) -> float:
    sequence_length = len(sequence)

    x_loop = 0.5
    total = 0

    for i in range(max_iterations):
        r_loop = input_x if (i % sequence_length) % 2 == 0 else input_y
        total += np.log(r_loop * (1 - 2 * x_loop))
        x_loop = r_loop * x_loop * (1 - x_loop)

    return total / max_iterations


class Lyapunov(EscapeFractal):
    def __init__(self, max_iterations: int, sequence: str):
        super().__init__(
            max_iterations=max_iterations,
            base_function=lyapunov_escape,
            jit_function=lyapunov_escape_jit,
        )
        self.sequence = np.array([1 if char == "A" else 0 for char in sequence])

    def _make_escape_function(self, mode):
        max_iterations = self.max_iterations
        sequence = self.sequence

        if mode == "normal":

            def bound_escape_function(input_x: float, input_y: float) -> float:
                return self.base_function(input_x, input_y, max_iterations, sequence)

        else:
            escape_function = self.jit_function

            @njit
            def bound_escape_function(input_x: float, input_y: float) -> float:
                return escape_function(input_x, input_y, max_iterations, sequence)

        return bound_escape_function

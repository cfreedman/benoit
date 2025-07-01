from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

from src.ascii import get_ascii
from src.grid import ComplexPoint, GridHalfSteps, GridPoints
from src.mandelbrot import EscapeFractal, compute_escape


class Render:
    def __init__(self, fractal: EscapeFractal, grid: GridPoints):
        self.grid = grid
        self.fractal = fractal

    def render_cpu(self):
        start = time.time()
        x_step, y_step = self.grid.steps()
        half_steps = GridHalfSteps(x_step / 2, y_step / 2)

    def render_parallel(self):
        start = time.time()
        x_step, y_step = self.grid.steps()
        half_steps = GridHalfSteps(x_step / 2, y_step / 2)

        centers = []
        for j in range(len(self.grid.y_grid)):
            for i in range(len(self.grid.x_grid)):
                x = self.grid.x_matrix[i, j]
                y = self.grid.y_matrix[i, -(j + 1)]
                center = ComplexPoint(x, y)

                centers.append(center)

        max_iterations = self.fractal.max_iterations
        partial_compute_escape = partial(
            compute_escape, half_steps=half_steps, fractal=self.fractal
        )

        with ProcessPoolExecutor() as executor:
            results = executor.map(partial_compute_escape, centers)

        results = list(results)
        width = len(self.grid.x_grid)
        height = len(self.grid.y_grid)

        for j in range(height):
            line = "".join(
                [
                    get_ascii(result, max_iterations)
                    for result in results[j * width : (j + 1) * width]
                ]
            )
            print(line)

        total_time = time.time() - start
        print(total_time)

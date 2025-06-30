from src.ascii import get_ascii
from src.grid import ComplexPoint, GridHalfSteps, GridPoints, generate_sample_grid
from src.mandelbrot import EscapeFractal


class Render:
    def __init__(self, fractal: EscapeFractal, grid: GridPoints):
        self.grid = grid
        self.fractal = fractal

    def render(self):
        x_step, y_step = self.grid.steps()
        half_steps = GridHalfSteps(x_step / 2, y_step / 2)

        for j in range(len(self.grid.y_grid)):
            line = ""
            for i in range(len(self.grid.x_grid)):
                x = self.grid.x_matrix[i, j]
                y = self.grid.y_matrix[i, -(j + 1)]
                center = ComplexPoint(x, y)

                sample_grid = generate_sample_grid(
                    center=center, half_steps=half_steps, x_divisions=5, y_divisions=5
                )

                average_escape = self.fractal.average_escape(sample_grid)
                character = get_ascii(average_escape, self.fractal.max_iterations)
                line += character

            print(line)

import numpy as np
from src.grid import (
    Box,
    CharacterBlock,
    ComplexPoint,
    Interval,
    SampleSize,
    generate_grid,
)
from src.julia import average_escape_julia
from src.mandelbrot import average_escape


palette = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]


def get_ascii(value: int, max_iterations: int) -> str:
    index = (
        int((value / max_iterations) * len(palette))
        if value < max_iterations
        else len(palette) - 1
    )
    return palette[index]


def mandelbrot():
    x_interval = Interval(-2, -1.5)
    y_interval = Interval(-0.1, 0.1)
    image_box = Box(x_interval, y_interval)
    grid, grid_halfsteps = generate_grid(image_box, 100)

    sample_size = SampleSize(x=5, y=5)

    xv, yv = np.meshgrid(grid.x_grid, grid.y_grid, indexing="ij")

    # lines = []

    for j in range(len(grid.y_grid)):
        line = ""
        for i in range(len(grid.x_grid)):
            center = ComplexPoint(
                xv[i, j], yv[i, -(j + 1)]
            )  # Indexing y_grid to select highest value first
            character_block = CharacterBlock(
                center=center, grid_halfsteps=grid_halfsteps, sample_size=sample_size
            )

            sample_grid = character_block.generate_sample_grid()

            average_escape_iteration = average_escape(
                grid=sample_grid, max_iterations=1000
            )

            character = get_ascii(value=average_escape_iteration, max_iterations=1000)
            line += character

        print(line)

        # lines.append(line)


def julia():
    x_interval = Interval(-2, 2)
    y_interval = Interval(-2, 2)
    image_box = Box(x_interval, y_interval)
    grid, grid_halfsteps = generate_grid(image_box, 100)

    sample_size = SampleSize(x=5, y=5)

    xv, yv = np.meshgrid(grid.x_grid, grid.y_grid, indexing="ij")

    for j in range(len(grid.y_grid)):
        line = ""
        for i in range(len(grid.x_grid)):
            center = ComplexPoint(
                xv[i, j], yv[i, -(j + 1)]
            )  # Indexing y_grid to select highest value first
            character_block = CharacterBlock(
                center=center, grid_halfsteps=grid_halfsteps, sample_size=sample_size
            )

            sample_grid = character_block.generate_sample_grid()

            c = ComplexPoint(0.35, 0.35)

            average_escape_iteration = average_escape_julia(
                grid=sample_grid, c=c, max_iterations=1000
            )

            character = get_ascii(value=average_escape_iteration, max_iterations=1000)
            line += character

        print(line)

        # lines.append(line)


if __name__ == "__main__":
    julia()

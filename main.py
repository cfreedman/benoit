import numpy as np
from src.grid import Box, Interval, generate_grid


palette = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]


def get_ascii(value: int, max_iteration: int) -> str:
    index = (value / max_iteration) * (len(palette) - 1)
    return palette[index]


def main():
    x_interval = Interval(-2, 2)
    y_interval = Interval(-1.5, 1.5)
    image_box = Box(x_interval, y_interval)
    grid, grid_halfsteps = generate_grid(image_box, 100)

    xv, yv = np.meshgrid(grid.x_grid, grid.y_grid, indexing="ij")

    lines = []

    for i in range(len(grid.x_grid)):
        for j in range(len(grid.x_grid)):


if __name__ == "__main__":
    main()

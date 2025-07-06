from dataclasses import dataclass
from typing import Tuple

import numpy as np

CHARACTER_ASPECT_RATIO = 2


@dataclass
class ComplexPoint:
    x: float
    y: float

    def __mul__(self, other: "ComplexPoint") -> "ComplexPoint":
        x = self.x * other.x - self.y * other.y
        y = self.x * other.y + self.y * other.x

        return ComplexPoint(x, y)

    def __add__(self, other: "ComplexPoint") -> "ComplexPoint":
        x = self.x + other.x
        y = self.y + other.y

        return ComplexPoint(x, y)

    def length_squared(self) -> float:
        return self.x**2 + self.y**2


@dataclass
class GridPoints:
    x_grid: np.ndarray
    y_grid: np.ndarray

    def steps(self) -> Tuple[float, float]:
        # Assumes that there are at least two elements in each grid
        x_step = self.x_grid[1] - self.x_grid[0]
        y_step = self.y_grid[1] - self.y_grid[0]

        return x_step, y_step


def generate_grid(
    x_center: float,
    y_center: float,
    x_length: float,
    y_length: float,
    x_divisions: int = 5,
    y_divisions: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    x_half_length, y_half_length = x_length / 2, y_length / 2

    x_min, y_min = x_center - x_half_length, y_center - y_half_length
    x_max, y_max = x_center + x_half_length, y_center + y_half_length

    x_unit, y_unit = (x_max - x_min) / x_divisions, (y_max - y_min) / y_divisions

    x_start, y_start = x_min + x_unit / 2, y_min + y_unit / 2
    x_end, y_end = x_max - x_unit / 2, y_max - y_unit / 2

    x_grid = np.linspace(x_start, x_end, x_divisions)
    y_grid = np.linspace(y_start, y_end, y_divisions)

    return x_grid, y_grid


def generate_square_grid(
    center: ComplexPoint, side_length: float, divisions: int
) -> GridPoints:
    # Adjust for higher line height than single character width?
    # height = math.ceil(aspect_ratio * width / CHARACTER_ASPECT_RATIO)

    (
        x_grid,
        y_grid,
    ) = generate_grid(
        x_center=center.x,
        y_center=center.y,
        x_length=side_length,
        y_length=side_length,
        x_divisions=divisions,
        y_divisions=int(divisions / 2),
    )

    return GridPoints(x_grid, y_grid)

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
class Interval:
    min_value: float
    max_value: float

    def length(self):
        return self.max_value - self.min_value


@dataclass
class Box:
    x_interval: Interval
    y_interval: Interval

    def aspect_ratio(self):
        return self.y_interval.length() / self.x_interval.length()


@dataclass
class GridPoints:
    x_grid: np.ndarray
    y_grid: np.ndarray

    x_matrix: np.ndarray
    y_matrix: np.ndarray

    def steps(self) -> Tuple[float, float]:
        # Assumes that there are at least two elements in each grid
        x_step = self.x_grid[1] - self.x_grid[0]
        y_step = self.y_grid[1] - self.y_grid[0]

        return x_step, y_step


@dataclass
class GridHalfSteps:
    x: float
    y: float


def generate_square_grid(
    start: ComplexPoint, size: float, divisions: int
) -> GridPoints:
    # Adjust for higher line height than single character width?
    # height = math.ceil(aspect_ratio * width / CHARACTER_ASPECT_RATIO)

    x_interval = Interval(start.x, start.x + size)
    y_interval = Interval(start.y, start.y + size)

    x_unit_width = x_interval.length() / divisions
    y_unit_width = y_interval.length() / (
        CHARACTER_ASPECT_RATIO * divisions
    )  # Compensate for characters having twice the height of width

    x_halfstep_width, y_halfstep_width = x_unit_width / 2, y_unit_width / 2

    x_grid = np.linspace(
        start=x_interval.min_value + x_halfstep_width,
        stop=x_interval.max_value - x_halfstep_width,
        num=divisions,
    )

    y_grid = np.linspace(
        start=y_interval.min_value + y_halfstep_width,
        stop=y_interval.max_value - y_halfstep_width,
        num=int(divisions / CHARACTER_ASPECT_RATIO),
    )

    x_matrix, y_matrix = np.meshgrid(x_grid, y_grid, indexing="ij")

    grid = GridPoints(x_grid, y_grid, x_matrix, y_matrix)

    return grid


def generate_sample_grid(
    center: ComplexPoint,
    half_steps: GridHalfSteps,
    x_divisions: int = 5,
    y_divisions: int = 5,
) -> GridPoints:
    x_interval = Interval(center.x - half_steps.x, center.x + half_steps.x)
    y_interval = Interval(center.y - half_steps.y, center.y + half_steps.y)

    x_unit_width = x_interval.length() / x_divisions
    y_unit_width = y_interval.length() / y_divisions

    x_halfstep_width, y_halfstep_width = x_unit_width / 2, y_unit_width / 2

    x_grid = np.linspace(
        start=x_interval.min_value + x_halfstep_width,
        stop=x_interval.max_value - x_halfstep_width,
        num=x_divisions,
    )

    y_grid = np.linspace(
        start=y_interval.min_value + y_halfstep_width,
        stop=y_interval.max_value - y_halfstep_width,
        num=y_divisions,
    )

    x_matrix, y_matrix = np.meshgrid(x_grid, y_grid, indexing="ij")

    grid = GridPoints(x_grid, y_grid, x_matrix, y_matrix)

    return grid

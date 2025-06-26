from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
        y = self.x + other.y

        return ComplexPoint(x, y)

    def length_squared(self) -> float:
        return self.x**2 + self.y**2


@dataclass
class Interval:
    min_value: float
    max_value: float

    def __len__(self):
        return self.max_value - self.min_value


@dataclass
class Box:
    x_interval: Interval
    y_interval: Interval

    def aspect_ratio(self):
        return len(self.y_interval) / len(self.x_interval)


@dataclass
class GridPoints:
    x_grid: np.ndarray
    y_grid: np.ndarray


@dataclass
class GridHalfSteps:
    x_halfstep: float
    y_halfstep: float


def generate_grid(box: Box, character_width: int) -> Tuple[GridPoints, GridHalfSteps]:
    aspect_ratio = box.aspect_ratio()

    # Adjust for higher line height than single character width?
    character_height = int(aspect_ratio * character_width)

    x_unit_width = len(box.x_interval) / character_width
    y_unit_width = len(box.y_interval) / character_height

    x_halfstep_width, y_halfstep_width = x_unit_width / 2, y_unit_width / 2

    x_grid = np.linspace(
        start=box.x_interval.min_value + x_halfstep_width,
        stop=box.x_interval.max_value - x_halfstep_width,
        step=character_width,
    )

    y_grid = np.linspace(
        start=box.y_interval.min_value + y_halfstep_width,
        stop=box.y_interval.max_value - y_halfstep_width,
        step=character_height,
    )

    grid = GridPoints(x_grid, y_grid)
    grid_halfsteps = GridHalfSteps(x_halfstep_width, y_halfstep_width)

    return grid, grid_halfsteps


@dataclass
class SampleSize:
    x: int
    y: int


@dataclass
class CharacterBlock:
    center: ComplexPoint
    grid_halfsteps: GridHalfSteps
    sample_size: SampleSize

    def generate_sample_grid(self):
        x_start, x_end = (
            self.center.x - self.grid_halfsteps.x_halfstep,
            self.center.x + self.grid_halfsteps.x_halfstep,
        )

        x_sample_step = (x_end - x_start) / self.sample_size.x

        x_grid = np.linspace(
            start=x_start + x_sample_step / 2,
            stop=x_end - x_sample_step / 2,
            step=self.sample_size.x,
        )

        y_start, y_end = (
            self.center.y - self.grid_halfsteps.y_halfstep,
            self.center.y + self.grid_halfsteps.y_halfstep,
        )

        y_sample_step = (y_end - y_start) / self.sample_size.y

        y_grid = np.linspace(
            start=y_start + y_sample_step / 2,
            stop=y_end - y_sample_step / 2,
            step=self.sample_size.y,
        )

        return GridPoints(x_grid, y_grid)

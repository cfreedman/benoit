import pytest

from src.grid import Box, Interval


def test_interval():
    interval = Interval(min_value=0, max_value=5)

    assert len(interval) == 5


def test_box():
    x_interval = Interval(min_value=0, max_value=5)
    y_interval = Interval(min_value=0, max_value=3)

    box = Box(x_interval, y_interval)

    print(box.aspect_ratio())

    assert box.aspect_ratio() == 3 / 5

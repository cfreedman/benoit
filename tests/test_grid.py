from src.grid import Box, ComplexPoint, Interval


def test_complex_point():
    a = ComplexPoint(1, 2)

    assert a * a == ComplexPoint(-3, 4)
    assert a + a == ComplexPoint(2, 4)
    assert a.length_squared() == 5


def test_interval():
    interval = Interval(min_value=0, max_value=5)

    assert interval.length() == 5


def test_box():
    x_interval = Interval(min_value=0, max_value=5)
    y_interval = Interval(min_value=0, max_value=3)

    box = Box(x_interval, y_interval)

    assert box.aspect_ratio() == 3 / 5

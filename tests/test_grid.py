from src.grid import ComplexPoint, generate_grid


def test_complex_point():
    a = ComplexPoint(1, 2)

    assert a * a == ComplexPoint(-3, 4)
    assert a + a == ComplexPoint(2, 4)
    assert a.length_squared() == 5


def test_generate_grid():
    x_grid, y_grid = generate_grid(0, 0, 2, 2, x_divisions=5, y_divisions=5)

    assert len(x_grid) == 5
    assert len(y_grid) == 5

    assert x_grid[0] == -0.8
    assert x_grid[-1] == 0.8

    assert y_grid[0] == -0.8
    assert y_grid[-1] == 0.8

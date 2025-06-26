from src.grid import ComplexPoint
from src.mandelbrot import escape


def test_mandelbrot_escape():
    c_in = ComplexPoint(0, 0)

    assert escape(c_in, max_iterations=1000) == 1000

    c_out = ComplexPoint(2, 0)

    assert escape(c_out, max_iterations=1000) < 1000

from numba.extending import is_jitted

from src.mandelbrot import (
    mandelbrot_escape,
    mandelbrot_escape_jit,
    mandelbrot_escape_gpu_jit,
    Mandelbrot,
)


def test_mandelbrot_escape():
    assert mandelbrot_escape(input_x=0, input_y=0, max_iterations=1000) == 1000

    assert mandelbrot_escape(input_x=2, input_y=0, max_iterations=1000) < 1000


def test_mandelbrot_escape_jit():
    assert mandelbrot_escape_jit(input_x=0, input_y=0, max_iterations=1000) == 1000

    assert mandelbrot_escape_jit(input_x=2, input_y=0, max_iterations=1000) < 1000


def test_mandelbrot_jit_selection():
    mandelbrot = Mandelbrot(max_iterations=1000)

    escape_function = mandelbrot.build_average_escape_function(mode="jit")

    assert is_jitted(escape_function)


def test_mandelbrot_class():
    mandelbrot = Mandelbrot(max_iterations=1000)

    # Test base escape function
    assert mandelbrot.base_function(0, 0, 1000) == 1000
    assert mandelbrot.base_function(2, 0, 1000) < 1000

    # Test JIT escape function
    assert mandelbrot.jit_function(0, 0, 1000) == 1000
    assert mandelbrot.jit_function(2, 0, 1000) < 1000

    # Test average escape function with normal mode
    average_escape_normal = mandelbrot.build_average_escape_function(mode="normal")
    assert average_escape_normal([0], [0]) == 1000

    # Test average escape function with JIT mode
    average_escape_jit = mandelbrot.build_average_escape_function(mode="jit")
    assert is_jitted(average_escape_jit)
    assert average_escape_jit([0], [0]) == 1000

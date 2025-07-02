from numba.extending import is_jitted
from numba.cuda.compiler import Dispatcher

from src.grid import ComplexPoint
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

    escape_function = mandelbrot.generate_escape_function(mode="jit")

    assert is_jitted(escape_function)


def test_mandelbrot_gpu_selection():
    mandelbrot = Mandelbrot(max_iterations=1000)

    escape_function = mandelbrot.generate_escape_function(mode="gpu")

    assert isinstance(escape_function, Dispatcher)

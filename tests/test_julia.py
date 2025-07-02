from numba.extending import is_jitted

from src.grid import ComplexPoint
from src.julia import (
    JuliaSet,
    julia_escape,
    julia_escape_jit,
    julia_escape_gpu,
)


def test_julia_escape():
    assert (
        julia_escape(
            input_x=0,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        == 1000
    )

    assert (
        julia_escape(
            input_x=2,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        < 1000
    )


def test_julia_escape_jit():
    assert (
        julia_escape_jit(
            input_x=0,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        == 1000
    )

    assert (
        julia_escape_jit(
            input_x=2,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        < 1000
    )


def test_julia_jit_selection():
    julia = JuliaSet(max_iterations=1000, parameter_x=0.35, parameter_y=0.35)

    escape_function = julia.generate_escape_function(mode="jit")

    assert is_jitted(escape_function)


def test_julia_gpu_selection():
    julia = JuliaSet(max_iterations=1000, parameter_x=0.35, parameter_y=0.35)

    escape_function = julia.generate_escape_function(mode="gpu")

    assert is_jitted(escape_function)

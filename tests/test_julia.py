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
    julia = JuliaSet(max_iterations=1000, parameter=ComplexPoint(0.35, 0.35))

    escape_function = julia.build_average_escape_function(mode="jit")

    assert is_jitted(escape_function)


def test_julia_class():
    julia = JuliaSet(max_iterations=1000, parameter=ComplexPoint(0.35, 0.35))

    # Test base escape function
    assert (
        julia.base_function(
            input_x=0,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        == 1000
    )
    assert (
        julia.base_function(
            input_x=2,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        < 1000
    )

    # Test JIT escape function
    assert (
        julia.jit_function(
            input_x=0,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        == 1000
    )
    assert (
        julia.jit_function(
            input_x=2,
            input_y=0,
            max_iterations=1000,
            parameter_x=0.35,
            parameter_y=0.35,
        )
        < 1000
    )

    # Test average escape function with normal mode
    average_escape_normal = julia.build_average_escape_function(mode="normal")
    assert average_escape_normal([0], [0]) == 1000

    # Test average escape function with JIT mode
    average_escape_jit = julia.build_average_escape_function(mode="jit")
    assert is_jitted(average_escape_jit)
    assert average_escape_jit([0], [0]) == 1000

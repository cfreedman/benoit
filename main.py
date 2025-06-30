from src.grid import (
    ComplexPoint,
    generate_square_grid,
)
from src.julia import JuliaSet
from src.mandelbrot import Mandelbrot
from src.render import Render


def mandelbrot():
    mandelbrot = Mandelbrot(max_iterations=1000)

    grid = generate_square_grid(start=ComplexPoint(-2, -2), size=4, divisions=100)
    renderer = Render(fractal=mandelbrot, grid=grid)

    renderer.render()


def julia():
    julia = JuliaSet(max_iterations=1000, parameter_point=ComplexPoint(0.35, 0.35))

    grid = generate_square_grid(start=ComplexPoint(-3, -3), size=6, divisions=50)
    renderer = Render(fractal=julia, grid=grid)

    renderer.render()


if __name__ == "__main__":
    mandelbrot()

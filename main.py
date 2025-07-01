import argparse

from src.grid import (
    ComplexPoint,
    GridPoints,
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

    grid = generate_square_grid(start=ComplexPoint(-1, -1), size=2, divisions=100)
    renderer = Render(fractal=julia, grid=grid)

    renderer.render()


def main():
    parser = argparse.ArgumentParser(
        prog="Benoit",
        description="ASCII rendering of common fractals and Lindenmayer systems",
    )

    parser.add_argument("fractal", choices=["mandelbrot", "julia"])
    parser.add_argument(
        "-c",
        "--parameter",
        nargs=2,
        type=float,
        help="The parameter point in complex space for defining the particular Julia set.",
    )

    parser.add_argument(
        "-p",
        "--point",
        nargs=2,
        type=float,
        default=(-2, -2),
        metavar=("X START", "Y START"),
        help="Specify the lower left hand corner point of the viewing box of the fractal in world space (x,y)",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=float,
        default=4,
        metavar="SIZE",
        help="Specify the length of the viewing box in world space along the x and y dimension",
    )
    parser.add_argument(
        "-d",
        "--divisions",
        type=int,
        default=100,
        metavar="DIVISIONS",
        help="The horizontal width of ASCII characters to render the fractal rendering i.e. the number of divisions of the horizontal interval in world space.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100,
        metavar="MAX ITERATIONS",
        help="The max number of iterations for the escape algorithm to run",
    )

    # args = parser.parse_args("mandelbrot --point -3 -1".split())
    # print(args)
    # parser.print_help()

    args = parser.parse_args()

    if args.fractal == "julia" and not args.parameter:
        parser.error("Parameter point --parameter is required for a Julia fractal.")

    fractal = (
        Mandelbrot(max_iterations=args.iterations)
        if args.fractal == "mandelbrot"
        else JuliaSet(
            max_iterations=args.iterations,
            parameter_point=ComplexPoint(x=args.parameter[0], y=args.parameter[1]),
        )
    )

    start_point = ComplexPoint(x=args.point[0], y=args.point[1])
    size = args.size
    divisions = args.divisions
    print(divisions)
    grid = generate_square_grid(start=start_point, size=size, divisions=divisions)

    renderer = Render(fractal=fractal, grid=grid)
    renderer.render_parallel()


if __name__ == "__main__":
    main()

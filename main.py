import argparse
import time

from numba.extending import is_jitted

from src.grid import (
    ComplexPoint,
    GridPoints,
    generate_grid,
    generate_square_grid,
)
from src.julia import JuliaSet
from src.mandelbrot import Mandelbrot, MandelbrotJIT
from src.render import Renderer


def main():
    parser = argparse.ArgumentParser(
        prog="Benoit",
        description="ASCII rendering of common fractals and Lindenmayer systems",
    )

    parser.add_argument("fractal", choices=["mandelbrot", "julia"])
    parser.add_argument(
        "-p",
        "--parameter",
        nargs=2,
        type=float,
        help="The parameter point in complex space for defining the particular Julia set.",
    )

    parser.add_argument(
        "-c",
        "--center",
        nargs=2,
        type=float,
        default=(0, 0),
        metavar=("X CENTER", "Y CENTER"),
        help="Specify the center point of the viewing box of the fractal in world space (x,y)",
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

    parser.add_argument(
        "-m", "--mode", choices=["normal", "jit", "gpu"], default="normal"
    )

    # args = parser.parse_args("mandelbrot --point -3 -1".split())
    # print(args)
    # parser.print_help()

    args = parser.parse_args()

    # if args.fractal == "julia" and not args.parameter:
    #     parser.error("Parameter point --parameter is required for a Julia fractal.")

    # fractal = (
    #     Mandelbrot(max_iterations=args.iterations)
    #     if args.fractal == "mandelbrot"
    #     else JuliaSet(
    #         max_iterations=args.iterations,
    #         parameter_x=args.parameter[0],
    #         parameter_y=args.parameter[1],
    #     )
    # )

    # center = ComplexPoint(x=args.center[0], y=args.center[1])
    # size = args.size
    # divisions = args.divisions
    # # grid = generate_square_grid(start=start_point, size=size, divisions=divisions)

    # grid = generate_square_grid(center=center, side_length=size, divisions=divisions)

    # renderer = Renderer(fractal=fractal, grid=grid, mode="normal")
    # renderer.render()

    fractal = Mandelbrot(max_iterations=1000)
    escape_function = fractal.build_escape_function(1000, mode="normal")

    print(is_jitted(escape_function))

    new_fractal = MandelbrotJIT(max_iterations=1000)
    new_escape_function = new_fractal.build_escape_function(1000, mode="jit")
    print(is_jitted(new_escape_function))

    # Warm up
    escape_function(0.0, 0.0)
    new_escape_function(0.0, 0.0)

    start = time.time()
    for i in range(100):
        escape_function(0.0, 0.0)
    print(time.time() - start)

    start = time.time()
    for i in range(100):
        new_escape_function(0.0, 0.0)
    print(time.time() - start)


if __name__ == "__main__":
    main()

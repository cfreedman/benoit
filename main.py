import argparse

from src.ascii import get_ascii
from src.grid import (
    ComplexPoint,
    generate_grid,
    generate_square_grid,
)
from src.julia import JuliaSet
from src.lyapunov import Lyapunov
from src.mandelbrot import Mandelbrot
from src.render import render


def main():
    parser = argparse.ArgumentParser(
        prog="Benoit",
        description="ASCII rendering of common fractals and Lindenmayer systems",
    )

    parser.add_argument("fractal", choices=["mandelbrot", "julia", "lyapunov"])
    parser.add_argument(
        "-p",
        "--parameter",
        nargs=2,
        type=float,
        help="The parameter point in complex space for defining the particular Julia set.",
    )
    parser.add_argument(
        "-q",
        "--sequence",
        type=str,
        help="The sequence parameter for calculating the Lyapunov fractal e.g. ABAABA",
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

    args = parser.parse_args()

    if args.fractal == "julia" and not args.parameter:
        parser.error("Parameter point --parameter is required for a Julia fractal.")
    if args.fractal == "lyapunov":
        if not args.sequence:
            parser.error("Sequence --sequence is required for a Lyapunov fractal.")

        x_center, y_center = args.center[0], args.center[1]
        size = args.size

        x_min, x_max = x_center - size / 2.0, x_center + size / 2.0
        y_min, y_max = y_center - size / 2.0, y_center + size / 2.0

        if x_min < 0 or y_min < 0 or x_max > 4 or y_max > 4:
            parser.error(
                "Viewing box for Lyapunov fractal must be contained with [0,4] x [0,4] region. Choose --center and --size inputs according to this restriction."
            )
    max_iterations = args.iterations
    mode = args.mode

    fractal = (
        Mandelbrot(max_iterations=args.iterations)
        if args.fractal == "mandelbrot"
        else JuliaSet(
            max_iterations=args.iterations,
            parameter=ComplexPoint(x=args.parameter[0], y=args.parameter[1]),
        )
        if args.fractal == "julia"
        else Lyapunov(max_iterations=args.iterations, sequence=args.sequence)
    )

    average_escape_function = fractal.build_average_escape_function(mode=mode)

    center = ComplexPoint(x=args.center[0], y=args.center[1])
    size = args.size
    divisions = args.divisions

    grid = generate_square_grid(center=center, side_length=size, divisions=divisions)

    results = render(
        x_grid=grid.x_grid,
        y_grid=grid.y_grid,
        sample_grid_function=generate_grid,
        average_escape_function=average_escape_function,
    )

    for j in range(len(grid.y_grid)):
        line = "".join(
            [
                get_ascii(value=value, max_iterations=max_iterations)
                for value in results[j, :]
            ]
        )
        print(line)


if __name__ == "__main__":
    main()

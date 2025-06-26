from math import sqrt


def calculate_boundary(c_x: float, c_y: float) -> float:
    """
    Solving for the valid Julia set radius satisfying R^2 - R - |c| >= 0
    """

    determinant = 1 + 4 * (c_x**2 + c_y**2)

    return (1 + sqrt(determinant)) / 2


def escape(z_x: float, z_y: float, c_x: float, c_y: float, max_iterations: int) -> int:
    boundary = calculate_boundary(c_x, c_y)

    z_x_loop = z_x
    z_y_loop = z_y
    iter = 0

    while z_x_loop**2 + z_y_loop**2 <= boundary**2 and iter <= max_iterations:
        z_x_temp = z_x_loop**2 + z_y_loop**2
        z_y_loop = 2 * z_x_loop * z_y_loop + c_y
        z_x_loop = z_x_temp + c_x

        iter += 1

    return iter

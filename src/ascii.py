palette = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]


def get_ascii(value: int, max_iterations: int) -> str:
    index = (
        int((value / max_iterations) * len(palette))
        if value < max_iterations
        else len(palette) - 1
    )
    return palette[index]

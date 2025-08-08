def colour_printer(msg, fg=None, bold=False):
    """
    Print a message with optional color and bold formatting using ANSI escape codes.
    Supported colors: black, red, green, yellow, blue, magenta, cyan, white.
    """
    colors = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
    }
    style = ""
    if bold:
        style += "\033[1m"
    if fg in colors:
        style += f"\033[{colors[fg]}m"
    print(f"{style}{msg}\033[0m")

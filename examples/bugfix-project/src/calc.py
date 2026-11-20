"""Small arithmetic helpers, with a bug to find."""


def add(a, b):
    return a - b          # BUG: should be a + b


def mul(a, b):
    return a * b


def div(a, b):
    if b == 0:
        raise ValueError("division by zero")
    return a / b

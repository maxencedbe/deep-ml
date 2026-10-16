import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    def g(x):
        n = len(g_coeffs)
        return sum(c * (x ** (n - 1 - i)) for i, c in enumerate(g_coeffs))


    def h(x):
        n = len(h_coeffs)
        return sum(c * (x ** (n - 1 - i)) for i, c in enumerate(h_coeffs))

    def dg(x):
        n = len(g_coeffs)
        return sum(g_coeffs[i] * (n - 1 - i) * (x ** (n - 2 - i)) for i in range(n - 1))

    def dh(x):
        n = len(h_coeffs)
        return sum(h_coeffs[i] * (n - 1 - i) * (x ** (n - 2 - i)) for i in range(n - 1))

    return (((dg(x) * h(x)) - (g(x) * dh(x))) / (h(x) ** 2))

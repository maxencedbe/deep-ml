import torch

def grad_of_quadratic(x_value: float) -> float:
    x = torch.tensor(x_value, dtype=torch.float, requires_grad=True)
    f = x ** 2 + 3 * x + 2
    f.backward()
    return float(x.grad)

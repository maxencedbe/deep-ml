import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    arr = torch.as_tensor(arr, dtype=torch.float32)

    if norm_type == "frobenius":
        if arr.ndim != 2:
            raise ValueError("`arr` must be 2D.")
        else:
            return float(torch.linalg.norm(arr, ord="fro"))
    
    arr = torch.ravel(arr)

    if norm_type == "l1":
        return float(torch.linalg.norm(arr, ord=1))

    elif norm_type == "l2":
        return float(torch.linalg.norm(arr, ord=2))

    elif norm_type == "linf":
        return float(torch.linalg.norm(arr, ord=torch.inf))

    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")

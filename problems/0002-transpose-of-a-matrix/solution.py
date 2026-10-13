import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a = torch.as_tensor(a)
    n, p = a.shape
    a_t = torch.zeros(p, n)
    for i in range(n):
        for j in range(p):
            a_t[j][i] = a[i][j]
    return a_t
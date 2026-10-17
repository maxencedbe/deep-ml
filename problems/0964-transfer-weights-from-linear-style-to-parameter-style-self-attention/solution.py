import numpy as np

def transfer_and_attend(X, W_q_lin, W_k_lin, W_v_lin):
    """
    Transfer linear-style weights (shape (d_out, d_in)) into parameter-style weights (shape (d_in, d_out)) and compute single-head causal self-attention.

    Args:
        X: input array of shape (seq_len, d_in)
        W_q_lin, W_k_lin, W_v_lin: linear-style weight matrices of shape (d_out, d_in)

    Returns:
        Nested list of shape (seq_len, d_out): the self-attention output.
    """
    X = np.array(X, dtype=float)
    W_q = np.array(W_q_lin, dtype=float).T
    W_k = np.array(W_k_lin, dtype=float).T
    W_v = np.array(W_v_lin, dtype=float).T

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    seq_len, d_in = X.shape
    d_out = Q.shape[1]

    S = (Q @ K.T) / (d_out ** 0.5)
    
    mask = np.triu(
        np.ones((seq_len, seq_len), dtype=bool),
        k=1
    )
    
    S = np.where(mask, -1e9, S)
    
    exp_S = np.exp(S - np.max(S, axis=-1, keepdims=True))
    attn_weights = exp_S / np.sum(exp_S, axis=-1, keepdims=True)
    
    output = attn_weights @ V

    return output.tolist()
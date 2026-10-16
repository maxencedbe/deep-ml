import numpy as np

def guidance_attention_mask(
    chunk_sizes: list[int],
    current_chunk: int
) -> np.ndarray:
    """
    Build a boolean attention mask for chunked autoregressive video generation.

    Args:
        chunk_sizes:   tokens per chunk, in order
        current_chunk: index of the chunk being generated

    Returns:
        Boolean array of shape (total_tokens, total_tokens).
        True means the row token can attend to the column token.
    """
    total_tokens = sum(chunk_sizes)
    history_tokens = sum(chunk_sizes[:current_chunk])
    current_tokens = chunk_sizes[current_chunk]
    current_end = history_tokens + current_tokens

    attention_mask = np.zeros((total_tokens, total_tokens), dtype=bool)

    attention_mask[:history_tokens, :history_tokens] = True
    attention_mask[history_tokens:current_end, :history_tokens] = True
    attention_mask[history_tokens:current_end, history_tokens:current_end] = np.tril(
        np.ones((current_tokens, current_tokens), dtype=bool)
    )

    return attention_mask
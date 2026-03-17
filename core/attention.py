"""Educational NumPy attention implementation used by unit tests."""

import numpy as np


def softmax(x):
    # Subtract max for numerical stability (log-sum-exp trick).
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(seq_len):
    # Upper triangle is future tokens, so those scores are masked out.
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    return np.where(mask, -1e9, 0.0)


def attention(Q, K, V):
    """
    Q, K, V: [n_heads, seq_len, d_head]
    returns: [n_heads, seq_len, d_head]
    """
    d_head = Q.shape[-1]
    seq_len = Q.shape[1]

    # [n_heads, seq_len, seq_len]
    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_head)
    scores = scores + causal_mask(seq_len)
    weights = softmax(scores)
    return weights @ V


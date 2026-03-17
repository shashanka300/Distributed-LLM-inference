# tests/test_attention.py
import numpy as np
from core.attention import attention, softmax

def test_output_shape():
    Q = np.random.randn(4, 8, 64)
    K = np.random.randn(4, 8, 64)
    V = np.random.randn(4, 8, 64)
    out = attention(Q, K, V)
    assert out.shape == (4, 8, 64), f"got {out.shape}"

def test_softmax_sums_to_one():
    x = np.random.randn(3, 5)
    s = softmax(x)
    np.testing.assert_allclose(s.sum(axis=-1), np.ones(3), atol=1e-6)

def test_causal_mask_blocks_future():
    # token 0 should produce zero attention weight toward token 1
    # Q, K identical and uniform  without mask both tokens equal weight
    Q = np.ones((1, 2, 4))
    K = np.ones((1, 2, 4))
    V = np.zeros((1, 2, 4))
    V[0, 0, 0] = 1.0   # token 0 has value 1 in dim 0
    V[0, 1, 0] = 99.0  # token 1 has value 99  should be invisible to token 0
    out = attention(Q, K, V)
    # token 0's output should be ~1.0, not influenced by token 1's 99
    assert out[0, 0, 0] < 5.0, "causal mask failed  future token leaked"

if __name__ == "__main__":
    test_output_shape()
    test_softmax_sums_to_one()
    test_causal_mask_blocks_future()
    print("all tests passed")



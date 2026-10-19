"""Check the hand-written backward pass against numerical gradients (float64)."""
import numpy as np
from gpt import GPT, softmax, cross_entropy

# stability checks
assert np.isfinite(softmax(np.array([[1000., 1001., 999.]]))).all()
l, _ = cross_entropy(np.array([[1000., 0., -1000.]]), np.array([2]))
assert np.isfinite(l) and abs(l - 2000) < 1e-6

rng = np.random.default_rng(1)
m = GPT(vocab=11, block=8, n_layer=2, n_head=2, n_embd=16, seed=3, dtype=np.float64)
for k in m.p:                       # randomize so no gradient is trivially zero
    if k.endswith(('_b', '.bqkv', '.bo', '.b1', '.b2')): m.p[k] += rng.standard_normal(m.p[k].shape) * 0.1
    if k.endswith('_g'): m.p[k] += rng.standard_normal(m.p[k].shape) * 0.1
    if m.p[k].ndim == 2: m.p[k] += rng.standard_normal(m.p[k].shape) * 0.1
x = rng.integers(0, 11, (3, 8)); y = rng.integers(0, 11, (3, 8))
_, grads = m.loss_and_grads(x, y)
worst = 0
for k, g in grads.items():
    flat = m.p[k].reshape(-1)
    for j in rng.choice(flat.size, min(5, flat.size), replace=False):
        o = flat[j]; e = 1e-6
        flat[j] = o + e; lp = m.loss(x, y)
        flat[j] = o - e; lm = m.loss(x, y)
        flat[j] = o
        num = (lp - lm) / (2 * e)
        rel = abs(num - g.reshape(-1)[j]) / max(1e-8, abs(num) + abs(g.reshape(-1)[j]))
        worst = max(worst, rel if abs(num) > 1e-7 else 0)
print(f"checked {len(grads)} parameter tensors, worst relative error {worst:.2e}")
assert worst < 1e-5, "gradient mismatch"
print("OK")

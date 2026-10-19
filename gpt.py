"""A tiny GPT in pure NumPy: forward AND backward written by hand. CPU only.

Pieces, in the order the video builds them:
  embedding -> softmax (stable) -> cross-entropy (stable) -> attention
  -> layer norm -> MLP -> transformer block -> full model -> Adam -> sampling
"""
import numpy as np

# ---------------------------------------------------------------- softmax / loss
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)      # subtract the max: avoids exp overflow
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def cross_entropy(logits, targets):
    """logits (N,V), targets (N,). Returns mean loss and dlogits.
    Uses log-softmax directly (logsumexp trick), so log(0) never happens."""
    z = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(z).sum(axis=-1, keepdims=True))
    logp = z - logsumexp
    N = logits.shape[0]
    loss = -logp[np.arange(N), targets].mean()
    d = np.exp(logp)                              # softmax probabilities
    d[np.arange(N), targets] -= 1.0               # d(loss)/d(logits) = p - onehot
    return loss, d / N

# ---------------------------------------------------------------- layers
def linear_fwd(x, W, b):
    return x @ W + b, (x, W)

def linear_bwd(dy, cache):
    x, W = cache
    dW = x.reshape(-1, x.shape[-1]).T @ dy.reshape(-1, dy.shape[-1])
    db = dy.reshape(-1, dy.shape[-1]).sum(0)
    return dy @ W.T, dW, db

def layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    return xhat * g + b, (xhat, rstd, g)

def layernorm_bwd(dy, cache):
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, xhat.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db

_K = np.sqrt(2.0 / np.pi)
def gelu_fwd(x):
    t = np.tanh(_K * (x + 0.044715 * x**3))
    return 0.5 * x * (1 + t), (x, t)

def gelu_bwd(dy, cache):
    x, t = cache
    dt = (1 - t**2) * _K * (1 + 3 * 0.044715 * x**2)
    return dy * (0.5 * (1 + t) + 0.5 * x * dt)

def attention_fwd(x, Wqkv, bqkv, Wo, bo, n_head):
    B, T, C = x.shape
    hd = C // n_head
    qkv, c1 = linear_fwd(x, Wqkv, bqkv)
    q, k, v = np.split(qkv, 3, axis=-1)
    sh = lambda a: a.reshape(B, T, n_head, hd).transpose(0, 2, 1, 3)   # (B,H,T,hd)
    q, k, v = sh(q), sh(k), sh(v)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    mask = np.triu(np.ones((T, T), bool), 1)
    s = np.where(mask, -1e9, s)                  # causal: can't look at the future
    a = softmax(s)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    y, c2 = linear_fwd(o, Wo, bo)
    return y, (c1, c2, q, k, v, a, n_head)

def attention_bwd(dy, cache):
    c1, c2, q, k, v, a, n_head = cache
    B, H, T, hd = q.shape
    do, dWo, dbo = linear_bwd(dy, c2)
    do = do.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
    da = do @ v.transpose(0, 1, 3, 2)
    dv = a.transpose(0, 1, 3, 2) @ do
    ds = a * (da - (da * a).sum(-1, keepdims=True))   # softmax backward
    ds = ds / np.sqrt(hd)                              # masked entries have a=0 -> ds=0
    dq = ds @ k
    dk = ds.transpose(0, 1, 3, 2) @ q
    mg = lambda a_: a_.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
    dqkv = np.concatenate([mg(dq), mg(dk), mg(dv)], axis=-1)
    dx, dWqkv, dbqkv = linear_bwd(dqkv, c1)
    return dx, dWqkv, dbqkv, dWo, dbo

# ---------------------------------------------------------------- model
class GPT:
    def __init__(self, vocab, block=64, n_layer=2, n_head=4, n_embd=64, seed=0, dtype=np.float32):
        self.V, self.T, self.L, self.H, self.C = vocab, block, n_layer, n_head, n_embd
        rng = np.random.default_rng(seed)
        C = n_embd
        nrm = lambda *s, std=0.02: (rng.standard_normal(s) * std).astype(dtype)
        zeros = lambda *s: np.zeros(s, dtype)
        ones = lambda *s: np.ones(s, dtype)
        p = {'wte': nrm(vocab, C), 'wpe': nrm(block, C),
             'lnf_g': ones(C), 'lnf_b': zeros(C), 'head': nrm(C, vocab)}
        for i in range(n_layer):
            rs = 0.02 / np.sqrt(2 * n_layer)       # scaled init for residual projections
            p.update({f'{i}.ln1_g': ones(C), f'{i}.ln1_b': zeros(C),
                      f'{i}.Wqkv': nrm(C, 3 * C), f'{i}.bqkv': zeros(3 * C),
                      f'{i}.Wo': nrm(C, C, std=rs), f'{i}.bo': zeros(C),
                      f'{i}.ln2_g': ones(C), f'{i}.ln2_b': zeros(C),
                      f'{i}.W1': nrm(C, 4 * C), f'{i}.b1': zeros(4 * C),
                      f'{i}.W2': nrm(4 * C, C, std=rs), f'{i}.b2': zeros(C)})
        self.p = p

    def n_params(self):
        return sum(v.size for v in self.p.values())

    def forward(self, idx):
        p, B, T = self.p, *idx.shape
        x = p['wte'][idx] + p['wpe'][:T]            # token + position embedding
        caches = []
        for i in range(self.L):
            h, c_ln1 = layernorm_fwd(x, p[f'{i}.ln1_g'], p[f'{i}.ln1_b'])     # pre-norm
            a, c_att = attention_fwd(h, p[f'{i}.Wqkv'], p[f'{i}.bqkv'], p[f'{i}.Wo'], p[f'{i}.bo'], self.H)
            x = x + a
            h, c_ln2 = layernorm_fwd(x, p[f'{i}.ln2_g'], p[f'{i}.ln2_b'])
            m, c_m1 = linear_fwd(h, p[f'{i}.W1'], p[f'{i}.b1'])
            m, c_g = gelu_fwd(m)
            m, c_m2 = linear_fwd(m, p[f'{i}.W2'], p[f'{i}.b2'])
            x = x + m
            caches.append((c_ln1, c_att, c_ln2, c_m1, c_g, c_m2))
        h, c_lnf = layernorm_fwd(x, p['lnf_g'], p['lnf_b'])
        logits, c_head = linear_fwd(h, p['head'], 0.0)
        return logits, (idx, caches, c_lnf, c_head)

    def backward(self, dlogits, cache):
        p = self.p
        idx, caches, c_lnf, c_head = cache
        g = {}
        dh, g['head'], _ = linear_bwd(dlogits, c_head)
        dx, g['lnf_g'], g['lnf_b'] = layernorm_bwd(dh, c_lnf)
        for i in reversed(range(self.L)):
            c_ln1, c_att, c_ln2, c_m1, c_g, c_m2 = caches[i]
            dm, g[f'{i}.W2'], g[f'{i}.b2'] = linear_bwd(dx, c_m2)
            dm = gelu_bwd(dm, c_g)
            dh, g[f'{i}.W1'], g[f'{i}.b1'] = linear_bwd(dm, c_m1)
            dl, g[f'{i}.ln2_g'], g[f'{i}.ln2_b'] = layernorm_bwd(dh, c_ln2)
            dx = dx + dl                                        # residual path
            da, g[f'{i}.Wqkv'], g[f'{i}.bqkv'], g[f'{i}.Wo'], g[f'{i}.bo'] = attention_bwd(dx, c_att)
            dl, g[f'{i}.ln1_g'], g[f'{i}.ln1_b'] = layernorm_bwd(da, c_ln1)
            dx = dx + dl
        g['wpe'] = np.zeros_like(p['wpe'])
        g['wpe'][:dx.shape[1]] = dx.sum(0)
        g['wte'] = np.zeros_like(p['wte'])
        np.add.at(g['wte'], idx, dx)
        return g

    def loss_and_grads(self, idx, targets):
        logits, cache = self.forward(idx)
        B, T, V = logits.shape
        loss, dl = cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        return loss, self.backward(dl.reshape(B, T, V), cache)

    def loss(self, idx, targets):
        logits, _ = self.forward(idx)
        return cross_entropy(logits.reshape(-1, self.V), targets.reshape(-1))[0]

    def generate(self, idx, n, temperature=1.0, top_k=None, rng=None):
        rng = rng or np.random.default_rng()
        idx = list(idx)
        for _ in range(n):
            ctx = np.array(idx[-self.T:])[None]
            logits = self.forward(ctx)[0][0, -1].astype(np.float64) / temperature
            if top_k:
                logits[logits < np.sort(logits)[-top_k]] = -1e9
            idx.append(int(rng.choice(self.V, p=softmax(logits))))
        return idx

class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.99, eps=1e-8, wd=0.0):
        self.lr, self.b1, self.b2, self.eps, self.wd, self.t = lr, b1, b2, eps, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = lr or self.lr
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= lr * (mh / (np.sqrt(vh) + self.eps) + self.wd * params[k])

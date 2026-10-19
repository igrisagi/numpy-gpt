"""Train the NumPy GPT on tiny Shakespeare (char level) and sample from it.
  python train.py                 # default ~10 min CPU run
  python train.py --steps 300     # quick smoke run
"""
import argparse, time, os, urllib.request, pickle
import numpy as np
from gpt import GPT, Adam

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--block', type=int, default=64)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--heads', type=int, default=4)
ap.add_argument('--embd', type=int, default=96)
ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data', default='input.txt')
ap.add_argument('--out', default='model.pkl')
a = ap.parse_args()

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
if not os.path.exists(a.data):
    urllib.request.urlretrieve(URL, a.data)
text = open(a.data, encoding='utf-8').read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
rng = np.random.default_rng(a.seed)

def batch(d):
    ix = rng.integers(0, len(d) - a.block - 1, a.batch)
    return (np.stack([d[i:i + a.block] for i in ix]),
            np.stack([d[i + 1:i + a.block + 1] for i in ix]))

model = GPT(len(chars), a.block, a.layers, a.heads, a.embd, seed=a.seed)
opt = Adam(model.p, lr=a.lr)
print(f"vocab {len(chars)}  params {model.n_params():,}  random-guess loss {np.log(len(chars)):.3f}")

def evaluate():
    return float(np.mean([model.loss(*batch(val)) for _ in range(10)]))

t0 = time.time()
for step in range(1, a.steps + 1):
    lr = a.lr * min(1, step / 50) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / a.steps)))
    x, y = batch(train)
    loss, grads = model.loss_and_grads(x, y)
    norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
    if norm > 1: grads = {k: g / norm for k, g in grads.items()}
    opt.step(model.p, grads, lr)
    if step % 50 == 0 or step == 1:
        extra = f"  val {evaluate():.3f}" if step % 250 == 0 or step == a.steps else ""
        print(f"step {step:5d}  train {loss:.3f}{extra}  {time.time() - t0:.0f}s", flush=True)

pickle.dump({'p': model.p, 'chars': chars, 'cfg': vars(a)}, open(a.out, 'wb'))
prompt = [stoi['\n']]
out = model.generate(prompt, 400, temperature=0.8, top_k=20, rng=np.random.default_rng(1))
print('\n--- sample ---\n' + ''.join(chars[i] for i in out))

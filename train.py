"""Train the NumPy GPT, char level. Two presets:

  python train.py                              # original demo: tiny Shakespeare, ~10 min
  python train.py --preset chat                # chatbot on chat.txt (run prepare_data.py first)
  python train.py --preset chat --minutes 180  # stop after 3 h; the checkpoint keeps progress
  python train.py --preset chat --resume       # continue from the last checkpoint

A checkpoint is saved to --out every --save-every steps and at the end, so a
long run can be stopped (or crash) and resumed without losing progress.
Any flag overrides its preset value, e.g. --steps 300 for a smoke run.
"""
import argparse, time, os, urllib.request, pickle
import numpy as np
from gpt import GPT, Adam

PRESETS = {
    # the original toy run: learns Shakespeare-ish text in ~10 min
    'shakespeare': dict(data='input.txt', block=64, layers=3, heads=4, embd=96,
                        steps=1500, batch=16, lr=3e-3),
    # the chatbot: bigger brain, movie dialogue, an evening of training
    'chat': dict(data='chat.txt', block=128, layers=4, heads=4, embd=128,
                 steps=20000, batch=8, lr=2.5e-3),
}

ap = argparse.ArgumentParser()
ap.add_argument('--preset', choices=sorted(PRESETS), default='shakespeare')
ap.add_argument('--steps', type=int)
ap.add_argument('--batch', type=int)
ap.add_argument('--block', type=int)
ap.add_argument('--layers', type=int)
ap.add_argument('--heads', type=int)
ap.add_argument('--embd', type=int)
ap.add_argument('--lr', type=float)
ap.add_argument('--minutes', type=float, help='stop after this many minutes (checkpoint is kept)')
ap.add_argument('--save-every', type=int, default=500)
ap.add_argument('--resume', action='store_true')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data')
ap.add_argument('--out', default='model.pkl')
a = ap.parse_args()

p = dict(PRESETS[a.preset])
for k in ('steps', 'batch', 'block', 'layers', 'heads', 'embd', 'lr', 'data'):
    if getattr(a, k) is not None:
        p[k] = getattr(a, k)

if a.preset == 'shakespeare' and not os.path.exists(p['data']):
    URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(URL, p['data'])
if not os.path.exists(p['data']):
    raise SystemExit(f"{p['data']} not found - run: python prepare_data.py")

text = open(p['data'], encoding='utf-8').read()
chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
rng = np.random.default_rng(a.seed)

def batch(d):
    ix = rng.integers(0, len(d) - p['block'] - 1, p['batch'])
    return (np.stack([d[i:i + p['block']] for i in ix]),
            np.stack([d[i + 1:i + p['block'] + 1] for i in ix]))

model = GPT(len(chars), p['block'], p['layers'], p['heads'], p['embd'], seed=a.seed)
opt = Adam(model.p, lr=p['lr'])
step0 = 0
if a.resume and os.path.exists(a.out):
    d = pickle.load(open(a.out, 'rb'))
    model.p = d['p']
    opt.m, opt.v, opt.t = d['opt']
    step0 = d['step']
    print(f"resumed from {a.out} at step {step0:,}")

print(f"preset {a.preset}  vocab {len(chars)}  params {model.n_params():,}  "
      f"random-guess loss {np.log(len(chars)):.3f}")

def evaluate():
    return float(np.mean([model.loss(*batch(val)) for _ in range(10)]))

def save(step):
    cfg = dict(p, preset=a.preset)
    pickle.dump({'p': model.p, 'chars': chars, 'cfg': cfg,
                 'opt': (opt.m, opt.v, opt.t), 'step': step}, open(a.out, 'wb'))

t0 = time.time()
budget = a.minutes * 60 if a.minutes else None
step = step0
for step in range(step0 + 1, p['steps'] + 1):
    lr = p['lr'] * min(1, step / 50) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / p['steps'])))
    x, y = batch(train)
    loss, grads = model.loss_and_grads(x, y)
    norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
    if norm > 1: grads = {k: g / norm for k, g in grads.items()}
    opt.step(model.p, grads, lr)
    el = time.time() - t0
    if step % 50 == 0 or step == step0 + 1:
        extra = f"  val {evaluate():.3f}" if step % 250 == 0 or step == p['steps'] else ""
        print(f"step {step:5d}  train {loss:.3f}{extra}  {el:.0f}s", flush=True)
    if step % a.save_every == 0:
        save(step)
    if budget and el > budget:
        print(f"time budget reached at step {step:,} - checkpoint saved; continue with --resume")
        break
save(step)

prompt = [stoi['\n']]
out = model.generate(prompt, 400, temperature=0.8, top_k=20, rng=np.random.default_rng(1))
print('\n--- sample ---\n' + ''.join(chars[i] for i in out))

"""Chat live with the model you trained:  python chat.py

Type a message, get a reply. 'quit' to stop.

Be fair to it: this brain is under a million parameters of pure NumPy,
trained from scratch on movie dialogue on a laptop CPU. It learned the
rhythm of conversation, not the internet. Short messages work best.
"""
import pickle
import numpy as np
from gpt import GPT

d = pickle.load(open('model.pkl', 'rb'))
c = d['cfg']; chars = d['chars']
stoi = {ch: i for i, ch in enumerate(chars)}
m = GPT(len(chars), c['block'], c['layers'], c['heads'], c['embd'])
m.p = d['p']
NL = stoi['\n']


def reply(question, seed):
    """The model only ever saw 'User: ...\nBot: ...' exchanges, so we ask it
    to complete exactly that pattern and stop it at the end of its line."""
    prompt = f"User: {question}\nBot:"
    ids = [stoi.get(ch, NL) for ch in prompt]
    out = m.generate(ids, 200, temperature=0.8, top_k=20,
                     rng=np.random.default_rng(seed), stop_ids={NL})
    return ''.join(chars[i] for i in out[len(ids):]).strip()


print(f"numpy-gpt chat - {m.n_params():,} parameters, trained from scratch on movie dialogue.")
print("It is not ChatGPT; it is yours. Short messages work best. 'quit' to stop.\n")
seed = 0
while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if q.lower() in ('quit', 'exit'):
        break
    if not q:
        continue
    seed += 1
    print("Bot:", reply(q, seed))

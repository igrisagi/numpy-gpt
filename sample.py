"""Generate text from a trained model.pkl:  python sample.py "ROMEO:" 300"""
import sys, pickle
import numpy as np
from gpt import GPT

d = pickle.load(open('model.pkl', 'rb'))
c = d['cfg']; chars = d['chars']; stoi = {ch: i for i, ch in enumerate(chars)}
m = GPT(len(chars), c['block'], c['layers'], c['heads'], c['embd']); m.p = d['p']
prompt = sys.argv[1] if len(sys.argv) > 1 else '\n'
n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
out = m.generate([stoi[ch] for ch in prompt], n, temperature=0.8, top_k=20)
print(''.join(chars[i] for i in out))

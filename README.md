# numpy-gpt

A small GPT in pure NumPy, now with a **live chat mode**. Forward and backward
passes are written by hand (no PyTorch, no autograd). Runs on CPU.

Built along the order of the "How I Built GPT from Scratch in Numpy" video:
embeddings, stable softmax, stable cross-entropy, causal attention, layer norm,
MLP, transformer blocks, full model, Adam, sampling. It is my own
implementation following that structure, not the video author's code.

## The honest part first

The chatbot is a ~830K-parameter character-level model trained from scratch on
movie dialogue, on a laptop CPU. That is about a million times smaller than
ChatGPT. What it actually learns:

- the *shape* of conversation: you type after `User:`, it answers after `Bot:`
- short, mostly grammatical replies in movie-dialogue style
- generic small talk, not facts. It does not know things; it knows how
  replies *sound*

That is the point of the project: a chatbot you trained yourself, end to end,
that you fully understand. No pretrained weights, no APIs, no magic.

## Chat mode (the new part)

```
pip install numpy
python prepare_data.py        # downloads the Cornell Movie Dialogs Corpus -> chat.txt
python train.py --preset chat # ~830K params; on an i5-10210U expect ~2-3 h for 20k steps
python chat.py                # talk to it
```

Training saves a checkpoint to `model.pkl` every 500 steps. Stop it any time
(Ctrl+C loses at most 500 steps) and continue later:

```
python train.py --preset chat --minutes 180   # train for 3 h, then stop cleanly
python train.py --preset chat --resume        # pick up where it left off
```

Longer training = better replies. 20k steps is the sweet spot for one evening;
loss keeps creeping down well past that if you let it run overnight.

### The data

[Cornell Movie Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html):
220,579 conversational exchanges between 10,292 movie characters in 617 films.
`prepare_data.py` keeps clean ASCII exchanges (120k of them, ~13 MB) and
formats each as:

```
User: <line A>
Bot: <line B>
```

The chat loop just feeds the model `User: your message\nBot:` and lets it
finish the line. Citation: Cristian Danescu-Niculescu-Mizil & Lillian Lee,
"Chameleons in imagined conversations", ACL 2011.

## Original Shakespeare demo (unchanged)

```
python test_gradients.py      # hand-written backward pass vs numerical gradients -> OK
python train.py               # char-level tiny Shakespeare, ~10 min
python sample.py "ROMEO:" 300
```

## Files
- `gpt.py` - the model: every layer's forward and backward, Adam, sampling
- `test_gradients.py` - checks the backward pass against numerical gradients
- `prepare_data.py` - downloads and formats the movie-dialogue corpus
- `train.py` - trains on either dataset (`--preset shakespeare` or `--preset chat`),
  with checkpoints, time budgets and resume
- `chat.py` - the live chat loop
- `sample.py` - generate text from a saved `model.pkl`
- `train_log.txt` - log of the original Shakespeare run

## Measured results (i5-10210U class laptop, no GPU)
- Gradient check: worst relative error 1.4e-6 over 29 parameter tensors (float64)
- Shakespeare model: 3 layers, 4 heads, 96 dims, context 64, 354,336 params;
  2500 steps, batch 16: loss 4.17 -> ~1.55 in 486 s
- Chat model: 4 layers, 4 heads, 128 dims, context 128, 833,280 params;
  ~3 steps/s at batch 8 on a 2-core sandbox (faster on the i5)

## Going bigger: the Kaggle GPU path

This code is CPU NumPy *by design* - the hand-written gradients are the lesson.
When you want a smarter bot, the free scaling path is:

1. **Kaggle** gives ~30 h/week of a free P100 GPU (kaggle.com, sign in, verify
   phone, open a Notebook with GPU accelerator).
2. **Minimal-change port:** swap `numpy` for [`cupy`](https://cupy.dev/)
   (`import cupy as np`). The hand-written forward/backward stays as-is and the
   arrays move to the GPU. Expect roughly 50-100x faster steps.
3. With that headroom the sensible next size is ~10-30M params
   (e.g. `--layers 8 --embd 256 --block 256`), context 256-512, and the full
   220k-exchange corpus. That is where replies start feeling genuinely
   on-topic - still not ChatGPT, but a real small chatbot.
4. Beyond that the honest move is a PyTorch port (autograd takes over the
   backward pass) and a BPE tokenizer instead of characters.

## Not done
No tokenizer (it is character level), no GPU support without the CuPy port,
no multi-turn memory beyond the context window.

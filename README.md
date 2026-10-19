# numpy-gpt

A small GPT in pure NumPy. Forward and backward passes are written by hand (no PyTorch, no autograd). Runs on CPU.

Built along the order of the "How I Built GPT from Scratch in Numpy" video: embeddings, stable softmax, stable cross-entropy, causal attention, layer norm, MLP, transformer blocks, full model, Adam, sampling. It is my own implementation following that structure, not the video author's code.

## Files
- `gpt.py` - the model, every layer's forward and backward, Adam, sampling
- `test_gradients.py` - checks the hand-written backward pass against numerical gradients
- `train.py` - character-level training on tiny Shakespeare (downloads the text if missing)
- `sample.py` - generate text from a saved `model.pkl`
- `train_log.txt` - log of the run below

## Run
```
pip install numpy
python test_gradients.py      # should print OK
python train.py               # ~8 min on a laptop CPU (2500 steps)
python sample.py "ROMEO:" 300
```

## Measured results (i5-10210U class laptop, no GPU)
- Gradient check: worst relative error 1.4e-6 over 29 parameter tensors (float64)
- Model: 3 layers, 4 heads, 96 dims, context 64, 354,336 parameters
- 2500 steps, batch 16: train loss 4.17 (random guess) -> ~1.55, validation loss 1.71, 486 s
- Output looks like Shakespeare-shaped text with real words and format but little meaning. That is expected at this size.

## Not done
No tokenizer (it is character level), no GPU support, no checkpoint resume.

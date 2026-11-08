import torch
import torch.nn as nn

from prepare_data import build_loaders

train_loader, val_loader, test_loader, vocab = (
    build_loaders()
)

xb, yb = next(iter(train_loader))

embedding = nn.Embedding(
    len(vocab),
    32,
    padding_idx=0
)

vectors = embedding(xb)

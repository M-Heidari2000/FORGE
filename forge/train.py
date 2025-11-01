import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from .models import MLPWithValueHead
from torch.utils.data import DataLoader


def map_pretrain_labels(
    labels: torch.Tensor,
):
    """
        randomly map the label to a number with the same parity
    """
    odds = [1, 3, 5, 7, 9]
    evens = [0, 2, 4, 6, 8]

    y = torch.tensor([
        np.random.choice(odds) if l%2 == 1 else np.random.choice(evens) for l in labels
    ], device=labels.device)

    return y


def pretrain_model(
    model: MLPWithValueHead,
    dataloader: DataLoader,
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, num_epochs+1)):
        epoch_losses = []
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y = map_pretrain_labels(y)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        print(f"epoch {epoch}, loss: {np.mean(epoch_losses)}")

    return model
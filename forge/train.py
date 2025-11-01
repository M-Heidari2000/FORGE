import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Literal
from .models import MLPWithValueHead
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils import (
    map_pretrain_labels,
    map_sft_labels,
    compute_mnist_rewards,
) 


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
            y = map_pretrain_labels(x, y)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        print(f"epoch {epoch}, loss: {np.mean(epoch_losses)}")

    return model


def finetune_sft(
    pretrained_model: MLPWithValueHead,
    dataloader: DataLoader,
    method: Literal["sft1", "sft2", "oracle"],
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
):
    
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)
    model.load_state_dict(copy.deepcopy(pretrained_model.state_dict()))

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
            y = map_sft_labels(y, method=method)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        print(f"epoch {epoch}, loss: {np.mean(epoch_losses)}")

    return model


def finetune_reinforce(
    pretrained_model: MLPWithValueHead,
    dataloader: DataLoader,
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,   
):
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)
    model.load_state_dict(copy.deepcopy(pretrained_model.state_dict()))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_rewards, epoch_steps = 0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, values = model(x)
            action_probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(action_probs, 1).squeeze(-1)

            rewards = compute_mnist_rewards(labels=y, actions=actions, method="reinforce")
            epoch_rewards += rewards.sum().item()
            epoch_steps += rewards.numel()

            action_log_probs = action_probs[torch.arange(len(actions)), actions].log()
            policy_loss = -(rewards * action_log_probs).mean()
            value_loss = F.mse_loss(values, rewards)
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch}, avg reward: {epoch_rewards / epoch_steps}")

    return model
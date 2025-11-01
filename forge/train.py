import copy
import torch
import wandb
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
    compute_kl,
    evaluate,
) 


def pretrain_model(
    model: MLPWithValueHead,
    pretrain_loader: DataLoader,
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
        for x, y in pretrain_loader:
            x, y = x.to(device), y.to(device)
            y = map_pretrain_labels(x, y)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        wandb.log({
            "pretraining/train/loss": np.mean(epoch_losses),
            "global_step": epoch,
        })

    return model


def finetune_sft(
    pretrained_model: MLPWithValueHead,
    finetune_loader: DataLoader,
    parity_test_loader: DataLoader,
    fashion_test_loader: DataLoader,
    method: Literal["sft1", "sft2", "oracle"],
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
    test_interval: int=2,
):
    
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)
    model.load_state_dict(copy.deepcopy(pretrained_model.state_dict()))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
        epoch_losses = []
        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device)
            y = map_sft_labels(y, method=method)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        wandb.log({
            f"sft {method}/train/loss": np.mean(epoch_losses),
            "global_step": epoch
        })

        if epoch % test_interval == 0:
            model.eval()
            pretrained_model.eval()
            parity_acc, fashion_acc = evaluate(
                model=model,
                parity_test_loader=parity_test_loader,
                fashion_test_loader=fashion_test_loader
            )
            forward_kl = compute_kl(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
                method="forward"
            )
            backward_kl = compute_kl(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
                method="backward"
            )
            wandb.log({
                f"sft {method}/test/parity_acc": parity_acc,
                f"sft {method}/test/fashion acc": fashion_acc,
                f"sft {method}/test/forward kl": forward_kl,
                f"sft {method}/test/backward kl": backward_kl,
                "global_step": epoch
            })

    return model


def finetune_reinforce(
    pretrained_model: MLPWithValueHead,
    finetune_loader: DataLoader,
    parity_test_loader: DataLoader,
    fashion_test_loader: DataLoader,
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
    test_interval: int=2,
):
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)
    model.load_state_dict(copy.deepcopy(pretrained_model.state_dict()))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        epoch_rewards, epoch_steps = 0, 0
        for x, y in finetune_loader:
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

        wandb.log({
            "reinforce/train/avg reward": epoch_rewards / epoch_steps,
            "global_step": epoch,
        })

        if epoch % test_interval == 0:
            model.eval()
            pretrained_model.eval()
            parity_acc, fashion_acc = evaluate(
                model=model,
                parity_test_loader=parity_test_loader,
                fashion_test_loader=fashion_test_loader
            )
            forward_kl = compute_kl(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
                method="forward"
            )
            backward_kl = compute_kl(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
                method="backward"
            )
            wandb.log({
                "reinforce/test/parity_acc": parity_acc,
                "reinforce/test/fashion acc": fashion_acc,
                "reinforce/test/forward kl": forward_kl,
                "reinforce/test/backward kl": backward_kl,
                "global_step": epoch,
            })

    return model
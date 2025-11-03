import copy
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Literal
from .models import MLPWithValueHead
from torch.utils.data import DataLoader
from .utils import (
    map_pretrain_labels,
    map_sft_labels,
    compute_mnist_rewards,
    compute_kl,
    evaluate,
    compute_tv,
) 


def pretrain_model(
    model: MLPWithValueHead,
    pretrain_loader: DataLoader,
    parity_test_loader: DataLoader,
    fashion_test_loader: DataLoader,
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
    test_interval: Optional[int]=2,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(1, num_epochs+1)):
        model.train()
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

        if epoch % test_interval == 0:
            model.eval()
            parity_acc, fashion_acc = evaluate(
                model=model,
                parity_test_loader=parity_test_loader,
                fashion_test_loader=fashion_test_loader
            )
            wandb.log({
                f"pretraining/test/parity accuracy": parity_acc,
                f"pretraining/test/fashion accuracy": fashion_acc,
                "global_step": epoch
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
    test_interval: Optional[int]=2,
):
    
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

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
            total_variation = compute_tv(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
            )

            wandb.log({
                f"sft {method}/test/parity accuracy": parity_acc,
                f"sft {method}/test/fashion accuracy": fashion_acc,
                f"sft {method}/test/forward kl": forward_kl,
                f"sft {method}/test/backward kl": backward_kl,
                f"sft {method}/test/total variation": total_variation,
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
    test_interval: Optional[int]=2,
):
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        epoch_rewards, epoch_steps = 0, 0
        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            action_probs = torch.softmax(logits, dim=1)
            actions = torch.multinomial(action_probs, 1).squeeze(-1)

            rewards = compute_mnist_rewards(labels=y, actions=actions, method="reinforce")
            epoch_rewards += rewards.sum().item()
            epoch_steps += rewards.numel()

            action_log_probs = action_probs[torch.arange(len(actions)), actions].log()
            policy_loss = -(rewards * action_log_probs).mean()

            optimizer.zero_grad()
            policy_loss.backward()
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
            total_variation = compute_tv(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
            )
            wandb.log({
                "reinforce/test/parity accuracy": parity_acc,
                "reinforce/test/fashion accuracy": fashion_acc,
                "reinforce/test/forward kl": forward_kl,
                "reinforce/test/backward kl": backward_kl,
                "reinforce/test/total variation": total_variation,
                "global_step": epoch,
            })

    return model


def finetune_ppo(
    pretrained_model: MLPWithValueHead,
    finetune_loader: DataLoader,
    parity_test_loader: DataLoader,
    fashion_test_loader: DataLoader,
    device: Optional[str]=None,
    num_epochs: Optional[int]=10,
    lr: Optional[float]=1e-3,
    test_interval: Optional[int]=2,
    eps: Optional[float]=0.2,
):
    # clone the pretrained model
    model = copy.deepcopy(pretrained_model)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        epoch_rewards, epoch_steps = 0, 0
        for x, y in finetune_loader:
            x, y = x.to(device), y.to(device)
            
            # sample trajectories with previous policy
            with torch.no_grad():
                old_logits, old_values = model(x)
                prev_action_dist = torch.softmax(old_logits, dim=1)
                actions = torch.multinomial(prev_action_dist, 1).squeeze(-1)
                rewards = compute_mnist_rewards(labels=y, actions=actions, method="reinforce")
                prev_action_probs = prev_action_dist[torch.arange(len(actions)), actions]
                advantages = rewards - old_values.squeeze(-1)

                epoch_rewards += rewards.sum().item()
                epoch_steps += rewards.numel()

            # compute loss and update policy and value networks
            logits, values = model(x)
            action_dist = torch.softmax(logits, dim=1)
            action_probs = action_dist[torch.arange(len(actions)), actions]
            prob_ratio = action_probs / prev_action_probs
            policy_loss = - torch.min(
                prob_ratio * advantages,
                torch.clamp(prob_ratio, 1-eps, 1+eps) * advantages
            ).mean()

            value_loss = F.mse_loss(values.squeeze(-1), rewards)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        wandb.log({
            "ppo/train/avg reward": epoch_rewards / epoch_steps,
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
            total_variation = compute_tv(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                dataloader=parity_test_loader,
            )
            wandb.log({
                "ppo/test/parity accuracy": parity_acc,
                "ppo/test/fashion accuracy": fashion_acc,
                "ppo/test/forward kl": forward_kl,
                "ppo/test/backward kl": backward_kl,
                "ppo/test/total variation": total_variation,
                "global_step": epoch,
            })

    return model
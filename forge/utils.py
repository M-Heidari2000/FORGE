import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .models import MLPWithValueHead
from typing import Literal



def map_pretrain_labels(
    data: torch.Tensor,
    labels: torch.Tensor,
):
    """
        randomly map the label to a number with the same parity
    """

    mnist_indexes = torch.nonzero(data[:, -1] == 1).flatten()

    odds = [1, 3, 5, 7, 9]
    evens = [0, 2, 4, 6, 8]

    y = labels.clone()

    for i in mnist_indexes:
        y[i] = np.random.choice(odds) if y[i] % 2 == 1 else np.random.choice(evens)

    return y


def map_sft_labels(
    labels: torch.Tensor,
    method: Literal["sft1", "sft2", "oracle"]
):
    """
        sft1: even labels mapped to 0 and odd digits to 1
        sft2: even digits randomly mapped to {0, 4} and odd digits to {1, 5}
        oracle: annotations drawn from the minimum-KL distribution consistent with task correctness
    """
    if method == "sft1":
        y = torch.tensor([0 if l%2 == 0 else 1 for l in labels], device=labels.device)
    elif method == "sft2":
        odds = [1, 5]
        evens = [0, 4]
        y = torch.tensor([
            np.random.choice(odds) if l%2 == 1 else np.random.choice(evens) for l in labels
        ], device=labels.device)
    elif method == "oracle":
        return NotImplementedError
    else:
        return ValueError("method must be one of [sft1, sft2, oracle]")
        
    return y


def compute_mnist_rewards(
    labels: torch.Tensor,
    actions: torch.Tensor,
    method: Literal["reinforce", "grpo"],
):
    """
        reinforce:
            +1 if parity is correct
            0 if parity is incorrect
        grpo:
            TODO
    """
    if method == "reinforce":
        # we need to detach the rewards
        rewards = (actions % 2 == labels % 2).float().detach()
    elif method == "grpo":
        raise NotImplementedError
    else:
        raise ValueError("method must be in [reinforce, grpo]")
    
    return rewards


def compute_kl(
    pretrained_model: MLPWithValueHead,
    finetuned_model: MLPWithValueHead,
    dataloader: DataLoader,
    method: Literal["forward", "backward"],
):
    device = next(finetuned_model.parameters()).device
    pretrained_model.eval()
    finetuned_model.eval()

    batch_kls = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            pretrained_logits, _ = pretrained_model(x)
            finetuned_logits, _ = finetuned_model(x)
            pretrained_log_probs = F.log_softmax(pretrained_logits, dim=1)
            finetuned_log_probs = F.log_softmax(finetuned_logits, dim=1)
            if method == "forward":
                kl = F.kl_div(
                    pretrained_log_probs,
                    finetuned_log_probs,
                    reduction="batchmean",
                    log_target=True,
                )
            elif method == "backward":
                kl = F.kl_div(
                    finetuned_log_probs,
                    pretrained_log_probs,
                    reduction="batchmean",
                    log_target=True,
                )
            else:
                raise ValueError("method must be in [backward, forward]")   
            batch_kls.append(kl.item())

    return np.mean(batch_kls)


def evaluate(
    model: MLPWithValueHead,
    parity_loader: DataLoader,
    fashion_loader: DataLoader,
):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # parity mnist test
        correct, total = 0, 0
        for x, y in tqdm(parity_loader):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred % 2 == y % 2).sum().item()
            total += x.shape[0]
        parity_acc = correct / total

        # fashion mnist test
        correct, total = 0, 0
        for x, y in tqdm(fashion_loader):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.shape[0]
        fashion_acc = correct / total

    return parity_acc, fashion_acc
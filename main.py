import torch
import argparse
import numpy as np
from forge.datasets import get_dataloaders
from forge.train import pretrain_model
from forge.models import MLPWithValueHead


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORGE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")
    parser.add_argument("--pretrain-n-epochs", type=int, default=10, help="number of epochs in pretraining")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3, help="learning rate for pretraining")
    parser.add_argument("--train-batch-size", type=int, default=64, help="batch size for training dataloaders")
    parser.add_argument("--test-batch-size", type=int, default=256, help="batch size for test dataloaders")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    pretrain_loader, fine_tune_loader, mnist_test_loader, fashion_test_loader = get_dataloaders(
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
    )

    base_model = MLPWithValueHead(in_features=785, out_features=10)
    pretrained_model = pretrain_model(
        model=base_model,
        dataloader=pretrain_loader,
        device=device,
        lr=args.pretrain_lr,
    )
    
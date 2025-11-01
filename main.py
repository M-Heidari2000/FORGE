import torch
import wandb
import argparse
import numpy as np
from forge.datasets import get_dataloaders
from forge.train import pretrain_model, finetune_sft, finetune_reinforce
from forge.models import MLPWithValueHead


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORGE")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--disable-gpu", action="store_true", default=False, help="disable using gpu")
    parser.add_argument("--pretrain-n-epochs", type=int, default=10, help="number of epochs in pretraining")
    parser.add_argument("--pretrain-lr", type=float, default=1e-3, help="learning rate for pretraining")
    parser.add_argument("--train-batch-size", type=int, default=64, help="batch size for training dataloaders")
    parser.add_argument("--test-batch-size", type=int, default=256, help="batch size for test dataloaders")
    parser.add_argument("--sft-n-epochs", type=int, default=10, help="number of epochs in sft")
    parser.add_argument("--sft-lr", type=float, default=1e-3, help="learning rate for sft")
    parser.add_argument("--reinforce-n-epochs", type=int, default=10, help="number of epochs in reinforce")
    parser.add_argument("--reinforce-lr", type=float, default=1e-4, help="learning rate for reinforce")
    parser.add_argument("--test-interval", type=int, default=2, help="test interval")

    args = parser.parse_args()

    wandb.init(
        project="RL project",
        name="RL's razor",
        config=vars(args),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = "cuda" if (torch.cuda.is_available() and not args.disable_gpu) else "cpu"

    pretrain_loader, finetune_loader, parity_test_loader, fashion_test_loader = get_dataloaders(
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
    )

    base_model = MLPWithValueHead(in_features=785, out_features=10)
    
    print("pretrain started ...")
    pretrained_model = pretrain_model(
        model=base_model,
        pretrain_loader=pretrain_loader,
        device=device,
        num_epochs=args.pretrain_n_epochs,
        lr=args.pretrain_lr,
    )

    print("sft1 finetuning started ...")
    sft1_model = finetune_sft(
        pretrained_model=pretrained_model,
        finetune_loader=finetune_loader,
        parity_test_loader=parity_test_loader,
        fashion_test_loader=fashion_test_loader,
        method="sft1",
        device=device,
        num_epochs=args.sft_n_epochs,
        lr=args.sft_lr,
        test_interval=args.test_interval,
    )
    
    print("sft2 finetuning started ...") 
    sft2_model = finetune_sft(
        pretrained_model=pretrained_model,
        finetune_loader=finetune_loader,
        parity_test_loader=parity_test_loader,
        fashion_test_loader=fashion_test_loader,
        method="sft2",
        device=device,
        num_epochs=args.sft_n_epochs,
        lr=args.sft_lr,
        test_interval=args.test_interval,
    )

    print("reinforce finetuning started ...")
    reinforce_model = finetune_reinforce(
        pretrained_model=pretrained_model,
        finetune_loader=finetune_loader,
        parity_test_loader=parity_test_loader,
        fashion_test_loader=fashion_test_loader,
        device=device,
        num_epochs=args.reinforce_n_epochs,
        lr=args.reinforce_lr,
        test_interval=args.test_interval,
    )
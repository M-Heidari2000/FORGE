import torch
from typing import Optional
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import (
    Dataset,
    Subset,
    ConcatDataset,
    DataLoader,
)


class CustomDataset(Dataset):
    def __init__(
        self,
        tv_dataset: Dataset,
        indicator: float,
    ):
        super().__init__()
        self.tv_dataset = tv_dataset
        self.indicator = indicator

    def __len__(self):
        return len(self.tv_dataset)
    
    def __getitem__(self, index):
        x, y = self.tv_dataset[index]
        x = torch.hstack((x.flatten(), torch.tensor(self.indicator)))

        return x, y
    

def get_dataloaders(
    train_batch_size: Optional[int]=64,
    test_batch_size: Optional[int]=256,     
    
):
    transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.5, ), (0.5, ))
    ])

    mnist_train = CustomDataset(
        tv_dataset=datasets.MNIST(root="./data", train=True, download=True, transform=transform),
        indicator=1.0
    )
    mnist_test = CustomDataset(
        tv_dataset=datasets.MNIST(root="./data", train=False, download=True, transform=transform),
        indicator=1.0
    )
    fashion_train = CustomDataset(
        tv_dataset=datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform),
        indicator=-1.0
    )
    fashion_test = CustomDataset(
        tv_dataset=datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform),
        indicator=-1.0
    )

    pretrain_subset = ConcatDataset([
        Subset(mnist_train, range(500)),
        Subset(fashion_train, range(500))
    ])

    finetune_subset = Subset(mnist_train, range(10000))

    pretrain_loader = DataLoader(pretrain_subset, batch_size=train_batch_size, shuffle=True)
    finetune_loader = DataLoader(finetune_subset, batch_size=train_batch_size, shuffle=True)
    parity_test_loader = DataLoader(mnist_test, batch_size=test_batch_size, shuffle=False)
    fashion_test_loader = DataLoader(fashion_test, batch_size=test_batch_size, shuffle=False)

    return pretrain_loader, finetune_loader, parity_test_loader, fashion_test_loader
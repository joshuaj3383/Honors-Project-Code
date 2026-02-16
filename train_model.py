import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm


def set_seed(seed: int = 2026):
    """Reproducability"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(batch_size=128, dataset_size=50000, seed=2026):
    """Loads datasets with given batch size"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    # Select a random portion of the training dataset to control training size
    if dataset_size is not None:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(train_dataset))[:dataset_size]
        train_dataset = Subset(train_dataset, indices)

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Runs through the dataset a single time and report information"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evalutes model on test dataset and reports statistics"""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


"""
Calculation functions
"""

# N: Model Parameters
def count_parameters(model):
    """Counts the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# C: Total Compute
def total_compute(N, D, epochs):
    """Note that although C is traditionally FLOPs, we can approximate it in this way"""
    return N * D * epochs


"""
=======Model Definition=======
Structure:
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class CNN(nn.Module):
    def __init__(self, width=32, blocks_per_stage=2):
        super().__init__()

        self.stage1 = self._make_stage(3, width, blocks_per_stage)
        self.stage2 = self._make_stage(width, 2 * width, blocks_per_stage)
        self.stage3 = self._make_stage(2 * width, 4 * width, blocks_per_stage)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4 * width, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)

        x = self.stage2(x)
        x = self.pool(x)

        x = self.stage3(x)

        # Global average pooling
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        """Private function for creating a stage of CNN blocks before pooling."""
        layers = [ConvBlock(in_channels, out_channels)]

        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels))

        return nn.Sequential(*layers)


def build_model(width, depth, device):
    model = CNN(width, depth).to(device)
    return model


def build_optimizer(model, num_epochs):
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
    )

    return optimizer, scheduler


def train_and_evaluate(model, train_loader, test_loader, optimizer, scheduler,
                       criterion, device, num_epochs, test_freq):
    final_loss_avg = 0
    final_acc_avg = 0
    best_test_loss = 999
    best_test_acc = 0
    epoch_records = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
        )

        if (epoch + 1) % test_freq == 0 or epoch >= num_epochs - 5:
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device
            )

            best_test_acc = max(best_test_acc, test_acc)
            best_test_loss = min(best_test_loss, test_loss)

            if epoch >= num_epochs - 5:
                final_loss_avg += test_loss
                final_acc_avg += test_acc

            print(
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
            )

        epoch_records.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
        })

    final_acc_avg /= 5
    final_loss_avg /= 5

    return {
        "final_loss_avg": final_loss_avg,
        "final_acc_avg": final_acc_avg,
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
        "epoch_records": epoch_records,
    }


def save_epoch_log(epoch_records, width, depth, seed):
    df = pd.DataFrame(epoch_records)
    epoch_log_path = f"results/runs/width{width}_depth{depth}_seed{seed}.csv"
    os.makedirs("results/runs", exist_ok=True)
    df.to_csv(epoch_log_path, index=False)


def append_summary(summary_row):
    summary_path = "results/final_data.csv"

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = pd.concat(
            [df_summary, pd.DataFrame([summary_row])],
            ignore_index=True
        )
    else:
        df_summary = pd.DataFrame([summary_row])

    df_summary.to_csv(summary_path, index=False)


def run_experiment(seed, batch_size, num_epochs, test_freq,
                   width, depth, dataset_size):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

    train_loader, test_loader = load_datasets(
        batch_size=batch_size,
        dataset_size=dataset_size,
        seed=seed
    )

    model = build_model(width, depth, device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model, num_epochs)

    results = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=num_epochs,
        test_freq=test_freq
    )

    save_epoch_log(results["epoch_records"], width, depth, seed)

    N = count_parameters(model)
    D = len(train_loader.dataset)

    summary_row = {
        "width": width,
        "depth": depth,
        "seed": seed,
        "N": N,
        "D": D,
        "C": total_compute(N, D, num_epochs),
        "epochs": num_epochs,
        "final_test_loss_avg_last5": results["final_loss_avg"],
        "final_test_acc_avg_last5": results["final_acc_avg"],
        "best_test_loss": results["best_test_loss"],
        "best_test_acc": results["best_test_acc"],
    }

    append_summary(summary_row)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--test_freq", type=int, default=5)

    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dataset_size", type=int, default=50000)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        test_freq=args.test_freq,
        width=args.width,
        depth=args.depth,
        dataset_size=args.dataset_size
    )

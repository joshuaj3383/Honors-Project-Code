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
from thop import profile


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# Dataset
# =========================================================

def load_datasets(batch_size=128, dataset_size=50000, seed=0):
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# =========================================================
# Model
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
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

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [ConvBlock(in_channels, out_channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.pool(x)
        x = self.stage2(x)
        x = self.pool(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(width, depth, device):
    return CNN(width, depth).to(device)


# =========================================================
# Training + Eval
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

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
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# =========================================================
# Checkpoint Utilities
# =========================================================

def get_checkpoint_dir(base_dir, width, depth, seed, dataset_size):
    path = os.path.join(
        base_dir,
        "checkpoints",
        f"width{width}_depth{depth}_seed{seed}_D{dataset_size}"
    )
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(state, checkpoint_dir, epoch):
    torch.save(state, os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"))
    torch.save(state, os.path.join(checkpoint_dir, "latest.pt"))


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest_path):
        return None

    checkpoint = torch.load(latest_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


# =========================================================
# Logging Utilities
# =========================================================

def log_epoch_to_csv(record, path):
    df = pd.DataFrame([record])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def append_summary(summary_row, base_dir):
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "final_data.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])

    df.to_csv(path, index=False)


# =========================================================
# Experiment
# =========================================================

def run_experiment(seed, batch_size, num_epochs, test_freq,
                   width, depth, dataset_size, base_dir):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_datasets(batch_size, dataset_size, seed)
    model = build_model(width, depth, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoint_dir = get_checkpoint_dir(base_dir, width, depth, seed, dataset_size)
    checkpoint = load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device)

    start_epoch = 0
    best_test_loss = float("inf")
    best_test_acc = 0
    final_loss_sum = 0
    final_acc_sum = 0
    epochs_for_avg = 0

    if checkpoint:
        print("Resuming from checkpoint.")
        start_epoch = checkpoint["epoch"] + 1
        best_test_loss = checkpoint["best_test_loss"]
        best_test_acc = checkpoint["best_test_acc"]
        final_loss_sum = checkpoint["final_loss_sum"]
        final_acc_sum = checkpoint["final_acc_sum"]
        epochs_for_avg = checkpoint["epochs_for_avg"]

    runs_dir = os.path.join(base_dir, "results", "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_csv_path = os.path.join(
        runs_dir,
        f"width{width}_depth{depth}_seed{seed}.csv"
    )

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        test_loss, test_acc = np.nan, np.nan

        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")

        if (epoch + 1) % test_freq == 0 or epoch >= num_epochs - min(5, num_epochs):

            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            best_test_loss = min(best_test_loss, test_loss)
            best_test_acc = max(best_test_acc, test_acc)
            print(f"Test  Loss: {test_loss:.4f} Test  Acc: {test_acc:.4f}")

        if epoch >= num_epochs - min(5, num_epochs):
            final_loss_sum += test_loss
            final_acc_sum += test_acc
            epochs_for_avg += 1

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        }

        log_epoch_to_csv(record, run_csv_path)

        save_checkpoint({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_test_loss": best_test_loss,
            "best_test_acc": best_test_acc,
            "final_loss_sum": final_loss_sum,
            "final_acc_sum": final_acc_sum,
            "epochs_for_avg": epochs_for_avg,
        }, checkpoint_dir, epoch)

    final_loss_avg = final_loss_sum / epochs_for_avg
    final_acc_avg = final_acc_sum / epochs_for_avg

    summary_row = {
        "width": width,
        "depth": depth,
        "seed": seed,
        "D": len(train_loader.dataset),
        "epochs": num_epochs,
        "final_test_loss_avg_last5": final_loss_avg,
        "final_test_acc_avg_last5": final_acc_avg,
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
    }

    append_summary(summary_row, base_dir)




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--test_freq", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dataset_size", type=int, default=50000)
    parser.add_argument("--base_dir", type=str, default="")

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
        dataset_size=args.dataset_size,
        base_dir=args.base_dir
    )
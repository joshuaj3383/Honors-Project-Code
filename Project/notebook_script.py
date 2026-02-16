from thop import profile
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
from tqdm import tqdm

# Paste above in notebook
#from google.colab import drive
#drive.mount('/content/drive')


BASE_DIR = "/content/drive/MyDrive/cifar_scaling"
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RUNS_DIR = os.path.join(RESULTS_DIR, "runs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_datasets(batch_size=128, dataset_size=50000, seed=2026):
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def proxy_compute(N, D, epochs):
    return N * D * epochs


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
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
        x = self.pool(self.stage1(x))
        x = self.pool(self.stage2(x))
        x = self.stage3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def _make_stage(self, in_c, out_c, num_blocks):
        layers = [ConvBlock(in_c, out_c)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(out_c, out_c))
        return nn.Sequential(*layers)


def build_model(width, depth, device):
    return CNN(width, depth).to(device)


def build_optimizer(model, num_epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return optimizer, scheduler


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    print(f"Resuming from epoch {checkpoint['epoch']}")
    return checkpoint["epoch"]


def evaluate(model, loader, criterion, device):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_and_evaluate(model, train_loader, test_loader,
                       optimizer, scheduler, criterion,
                       device, num_epochs, test_freq,
                       checkpoint_path, run_csv_path):

    best_test_loss = float("inf")
    best_test_acc = 0
    final_loss_avg = 0
    final_acc_avg = 0
    last_k = min(5, num_epochs)

    start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, leave=False):
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

        scheduler.step()

        train_loss = total_loss / total
        train_acc = correct / total

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        test_loss = np.nan
        test_acc = np.nan

        if (epoch + 1) % test_freq == 0 or epoch >= num_epochs - last_k:
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            best_test_acc = max(best_test_acc, test_acc)
            best_test_loss = min(best_test_loss, test_loss)

            if epoch >= num_epochs - last_k:
                final_loss_avg += test_loss
                final_acc_avg += test_acc

            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

        # Append per-epoch CSV
        epoch_row = pd.DataFrame([{
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        }])

        epoch_row.to_csv(
            run_csv_path,
            mode="a",
            header=not os.path.exists(run_csv_path),
            index=False
        )

        save_checkpoint(model, optimizer, scheduler, epoch + 1, checkpoint_path)

    final_loss_avg /= last_k
    final_acc_avg /= last_k

    return {
        "final_loss_avg": final_loss_avg,
        "final_acc_avg": final_acc_avg,
        "best_test_loss": best_test_loss,
        "best_test_acc": best_test_acc,
    }


def run_experiment(seed, batch_size, num_epochs, test_freq,
                   width, depth, dataset_size):

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = load_datasets(batch_size, dataset_size, seed)
    model = build_model(width, depth, device)

    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32).to(device)
    flops_forward, _ = profile(model, inputs=(dummy,), verbose=False)

    steps = len(train_loader) * num_epochs
    total_flops = 3 * flops_forward * steps

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer(model, num_epochs)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"width{width}_depth{depth}_D{dataset_size}_seed{seed}.pt"
    )

    run_csv_path = os.path.join(
        RUNS_DIR,
        f"width{width}_depth{depth}_D{dataset_size}_seed{seed}.csv"
    )

    results = train_and_evaluate(
        model, train_loader, test_loader,
        optimizer, scheduler, criterion,
        device, num_epochs, test_freq,
        checkpoint_path, run_csv_path
    )

    summary_row = {
        "width": width,
        "depth": depth,
        "seed": seed,
        "D": dataset_size,
        "epochs": num_epochs,
        "steps": steps,
        "total_flops": total_flops,
        "final_test_loss_avg_last5": results["final_loss_avg"],
        "final_test_acc_avg_last5": results["final_acc_avg"],
        "best_test_loss": results["best_test_loss"],
        "best_test_acc": results["best_test_acc"],
    }

    summary_path = os.path.join(RESULTS_DIR, "final_data.csv")

    pd.DataFrame([summary_row]).to_csv(
        summary_path,
        mode="a",
        header=not os.path.exists(summary_path),
        index=False
    )


if __name__ == "__main__":

    widths = [8, 16, 32, 64, 128]
    depths = [1, 2, 3]
    dataset_sizes = [6250, 12500, 25000, 50000]
    seeds = [0, 1, 2]

    batch_size = 512
    num_epochs = 100
    test_freq = 5

    summary_path = os.path.join(RESULTS_DIR, "final_data.csv")

    for seed in seeds:
        for width in widths:
            for depth in depths:
                for dataset_size in dataset_sizes:

                    if os.path.exists(summary_path):
                        df = pd.read_csv(summary_path)
                        already_done = (
                            (df["seed"] == seed) &
                            (df["width"] == width) &
                            (df["depth"] == depth) &
                            (df["D"] == dataset_size)
                        ).any()
                        if already_done:
                            print(f"Skipping completed: w={width}, d={depth}, D={dataset_size}")
                            continue

                    print(f"Running: seed={seed}, width={width}, depth={depth}, D={dataset_size}")

                    run_experiment(
                        seed=seed,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        test_freq=test_freq,
                        width=width,
                        depth=depth,
                        dataset_size=dataset_size
                    )

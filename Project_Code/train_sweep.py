from train_model import run_experiment
import os
import pandas as pd

widths = [8, 16, 32, 64, 128]
depths = [1, 2, 3]
dataset_sizes = [6250, 12500, 25000, 50000]
seeds = [0, 1, 2]

batch_size = 512
num_epochs = 100
test_freq = 5

summary_path = "results/final_data.csv"

os.makedirs("results", exist_ok=True)

for seed in seeds:
    for width in widths:
        for depth in depths:
            for dataset_size in dataset_sizes:

                # Skip completed runs
                if os.path.exists(summary_path):
                    df = pd.read_csv(summary_path)
                    already_done = (
                        (df["seed"] == seed) &
                        (df["width"] == width) &
                        (df["depth"] == depth) &
                        (df["D"] == dataset_size)
                    ).any()

                    if already_done:
                        print(f"Skipping: w={width}, d={depth}, D={dataset_size}, seed={seed}")
                        continue

                print(f"Running: w={width}, d={depth}, D={dataset_size}, seed={seed}")

                run_experiment(
                    seed=seed,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    test_freq=test_freq,
                    width=width,
                    depth=depth,
                    dataset_size=dataset_size
                )

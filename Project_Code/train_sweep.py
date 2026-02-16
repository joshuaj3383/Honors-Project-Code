from train_model import run_experiment
import os
import pandas as pd
import json


def sweep_train(
        widths,
        depths,
        dataset_sizes,
        seeds,
        batch_size,
        num_epochs,
        test_freq,
        summary_path
    ):

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

if __name__ == "__main__":
    widths = []
    depths = []
    dataset_sizes = []
    seeds = []

    batch_size = 512
    num_epochs = 100
    test_freq = 5

    summary_path = "results/final_data.csv"


    with open("Experiment_Parameters.json", "r") as f:
        data = json.load(f)

        for train_name in data:
            print("====================================================================================")
            print(f"Training Sweep: {train_name}")
            print("====================================================================================\n")
            train_info = data[train_name]
            widths = train_info["width"]
            depths = train_info["depth"]
            dataset_sizes = train_info["dataset_size"]
            seeds = train_info["seeds"]

            sweep_train(widths, depths, dataset_sizes, seeds, batch_size, num_epochs, test_freq, summary_path)
import os
import pandas as pd
import json
from train_model import run_experiment


####################################################################
#   Does sweep training through specified sets of parameters       #
#   Created by Joshua Johnston                                     #
#   BHCC Honors Project Spring 2026                                #
#   Writen for Notebook                                            #
####################################################################


def sweep_train(
        widths,
        depths,
        dataset_sizes,
        seeds,
        batch_size,
        num_epochs,
        test_freq,
        base_dir
    ):

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "final_data.csv")

    # Horrendous loop but it's a small script
    for seed in seeds:
        for width in widths:
            for depth in depths:
                for dataset_size in dataset_sizes:
                    # Check if done
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

                    # Train a model based on these parameters
                    run_experiment(
                        seed=seed,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        test_freq=test_freq,
                        width=width,
                        depth=depth,
                        dataset_size=dataset_size,
                        base_dir=base_dir
                    )


if __name__ == "__main__":

    # Google Drive Location
    BASE_DIR = "/content/drive/MyDrive/cifar_scaling"
    # JSON
    JSON_FILE = "Experiment_Parameters.json"

    # Constants
    batch_size = 512
    num_epochs = 100
    test_freq = 5

    # Train based on json file
    with open(os.path.join(BASE_DIR, JSON_FILE), "r") as f:
        data = json.load(f)

        for train_name in data:
            print("="*80)
            print(f"Training Sweep: {train_name}")
            print("="*80)

            train_info = data[train_name]

            sweep_train(
                train_info["width"],
                train_info["depth"],
                train_info["dataset_size"],
                train_info["seeds"],
                batch_size,
                num_epochs,
                test_freq,
                BASE_DIR
            )

import os
import pandas as pd
import json
from train_model import run_experiment, count_parameters, CNN


####################################################################
#   Varies model Width and number of epochs to keep fixed compute  #
#   Created by Joshua Johnston                                     #
#   BHCC Honors Project Spring 2026                                #
#   Writen for Notebook                                            #
####################################################################


def get_epochs(width: int, depth: int = 2, fixed_flops: int = 5748850000000, D: int = 50000,) -> int:
    test_model = CNN(width, depth)
    N = count_parameters(test_model)

    return int(fixed_flops / (N * D))

def train(
        widths,
        depth,
        dataset_size,
        seed,
        batch_size,
        test_freq,
        base_dir,
        fixed_flops,
    ):


    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "final_data.csv")

    df = None
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

    for width in widths:
        # Check if done
        if df is not None:
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
            num_epochs=get_epochs(width=width, depth=depth, fixed_flops=fixed_flops, D=dataset_size),
            test_freq=test_freq,
            width=width,
            depth=depth,
            dataset_size=dataset_size,
            base_dir=base_dir
        )


if __name__ == "__main__":
    # Google Drive Location
    BASE_DIR = "/content/drive/MyDrive/cifar_scaling"


    # Constants
    widths = range(16, 81, 8)
    depth = 2
    dataset_size = 50000
    seed = 0
    batch_size = 512
    test_freq = 5

    fixed_flops = 5748850000000

    train(
        widths,
        depth,
        dataset_size,
        seed,
        batch_size,
        test_freq,
        base_dir=BASE_DIR,
        fixed_flops=fixed_flops,
    )
















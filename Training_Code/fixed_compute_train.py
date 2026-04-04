import os
import pandas as pd
from train_model import run_experiment, count_parameters, proxy_compute, CNN

####################################################################
#   Varies model Width and number of epochs to keep fixed compute  #
#   Created by Joshua Johnston                                     #
#   BHCC Honors Project Spring 2026                                #
#   Writen for Notebook                                            #
####################################################################


def get_epochs(width: int, depth: int = 2, fixed_flops: int = 5748850000000, D: int = 50000,) -> int:
    test_model = CNN(width, depth)
    N = count_parameters(test_model)

    return max(1,round(fixed_flops / (N * D)))

def get_FLOPs(width: int, depth: int = 2, D: int = 50000, epochs: int = 50) -> int:
    test_model = CNN(width, depth)
    N = count_parameters(test_model)
    return proxy_compute(N,D,epochs)

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

    summary_path = os.path.join(results_dir, "fixed_compute_data.csv")

    df = None
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

    for width in widths:
        epochs = get_epochs(width, depth, fixed_flops, dataset_size)

        if df is not None:
            already_done = (
                    (df["seed"] == seed) &
                    (df["width"] == width) &
                    (df["depth"] == depth) &
                    (df["D"] == dataset_size) &
                    (df["epochs"] == epochs)
            ).any()

            if already_done:
                print(f"Skipping: w={width}, epochs={epochs}")
                continue

        actual_C = get_FLOPs(width, depth, dataset_size, epochs)
        print(f"Running: w={width}, epochs={epochs}, C≈{actual_C:.2e}")

        run_experiment(
            seed=seed,
            batch_size=batch_size,
            num_epochs=epochs,
            test_freq=test_freq,
            width=width,
            depth=depth,
            dataset_size=dataset_size,
            base_dir=base_dir
        )


if __name__ == "__main__":
    # Google Drive Location
    BASE_DIR = "/content/drive/MyDrive/cifar_compute"


    # Constants
    widths = range(32,57,8)
    depth = 2
    dataset_size = 50000
    seed = 1
    batch_size = 256
    test_freq = 10

    fixed_flops = get_FLOPs(32, depth, dataset_size, 50)

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

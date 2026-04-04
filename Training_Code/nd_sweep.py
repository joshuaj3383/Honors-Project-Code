from train_model import run_experiment

if __name__ == "__main__":
    widths = [8, 16, 32, 64]
    epochs = [25, 50, 75, 100]

    for width in widths:
        for epoch in epochs:
            print(f"Running experiment with width={width} and epochs={epoch}")
            run_experiment(
                seed=0,
                batch_size=256,
                num_epochs=epoch,
                test_freq=10,
                width=width,
                depth=2,
                dataset_size=50000,
                base_dir="/content/drive/MyDrive/cifar_scaling"
            )
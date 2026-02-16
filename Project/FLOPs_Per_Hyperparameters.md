# FLOPs Per Model Based on Hyperparameters

## Code:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    widths = [8, 16, 32, 64, 128]
    depths = [1, 2, 3]
    datasize = [6250,12500,25000,50000]

    batch_size = 512
    num_epochs = 1


    for dsize in datasize:
        train_loader, _ = load_datasets(batch_size=batch_size,dataset_size=dsize)
        print(f"Dataset Size: {dsize}")
        for width in widths:
            for depth in depths:
                model = build_model(width, depth, device)

                flops_forward, total_FLOPs, steps = count_FLOPs(
                    model=model,
                    train_loader=train_loader,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    device=device
                )

                print(f"Width: {width}, Depth: {depth}, Total FLOPs: {total_FLOPs:.3e}")


## Data:

    Dataset Size: 6250
    Width: 8, Depth: 1, Total FLOPs: 1.739e+10
    Width: 8, Depth: 2, Total FLOPs: 5.387e+10
    Width: 8, Depth: 3, Total FLOPs: 9.034e+10
    Width: 16, Depth: 1, Total FLOPs: 5.833e+10
    Width: 16, Depth: 2, Total FLOPs: 2.020e+11
    Width: 16, Depth: 3, Total FLOPs: 3.456e+11
    Width: 32, Depth: 1, Total FLOPs: 2.109e+11
    Width: 32, Depth: 2, Total FLOPs: 7.808e+11
    Width: 32, Depth: 3, Total FLOPs: 1.351e+12
    Width: 64, Depth: 1, Total FLOPs: 7.986e+11
    Width: 64, Depth: 2, Total FLOPs: 3.069e+12
    Width: 64, Depth: 3, Total FLOPs: 5.340e+12
    Width: 128, Depth: 1, Total FLOPs: 3.105e+12
    Width: 128, Depth: 2, Total FLOPs: 1.217e+13
    Width: 128, Depth: 3, Total FLOPs: 2.123e+13
    Dataset Size: 12500
    Width: 8, Depth: 1, Total FLOPs: 3.344e+10
    Width: 8, Depth: 2, Total FLOPs: 1.036e+11
    Width: 8, Depth: 3, Total FLOPs: 1.737e+11
    Width: 16, Depth: 1, Total FLOPs: 1.122e+11
    Width: 16, Depth: 2, Total FLOPs: 3.884e+11
    Width: 16, Depth: 3, Total FLOPs: 6.646e+11
    Width: 32, Depth: 1, Total FLOPs: 4.055e+11
    Width: 32, Depth: 2, Total FLOPs: 1.502e+12
    Width: 32, Depth: 3, Total FLOPs: 2.597e+12
    Width: 64, Depth: 1, Total FLOPs: 1.536e+12
    Width: 64, Depth: 2, Total FLOPs: 5.902e+12
    Width: 64, Depth: 3, Total FLOPs: 1.027e+13
    Width: 128, Depth: 1, Total FLOPs: 5.971e+12
    Width: 128, Depth: 2, Total FLOPs: 2.340e+13
    Width: 128, Depth: 3, Total FLOPs: 4.083e+13
    Dataset Size: 25000
    Width: 8, Depth: 1, Total FLOPs: 6.554e+10
    Width: 8, Depth: 2, Total FLOPs: 2.030e+11
    Width: 8, Depth: 3, Total FLOPs: 3.405e+11
    Width: 16, Depth: 1, Total FLOPs: 2.199e+11
    Width: 16, Depth: 2, Total FLOPs: 7.612e+11
    Width: 16, Depth: 3, Total FLOPs: 1.303e+12
    Width: 32, Depth: 1, Total FLOPs: 7.949e+11
    Width: 32, Depth: 2, Total FLOPs: 2.943e+12
    Width: 32, Depth: 3, Total FLOPs: 5.091e+12
    Width: 64, Depth: 1, Total FLOPs: 3.010e+12
    Width: 64, Depth: 2, Total FLOPs: 1.157e+13
    Width: 64, Depth: 3, Total FLOPs: 2.013e+13
    Width: 128, Depth: 1, Total FLOPs: 1.170e+13
    Width: 128, Depth: 2, Total FLOPs: 4.587e+13
    Width: 128, Depth: 3, Total FLOPs: 8.003e+13
    Dataset Size: 50000
    Width: 8, Depth: 1, Total FLOPs: 1.311e+11
    Width: 8, Depth: 2, Total FLOPs: 4.061e+11
    Width: 8, Depth: 3, Total FLOPs: 6.810e+11
    Width: 16, Depth: 1, Total FLOPs: 4.397e+11
    Width: 16, Depth: 2, Total FLOPs: 1.522e+12
    Width: 16, Depth: 3, Total FLOPs: 2.605e+12
    Width: 32, Depth: 1, Total FLOPs: 1.590e+12
    Width: 32, Depth: 2, Total FLOPs: 5.886e+12
    Width: 32, Depth: 3, Total FLOPs: 1.018e+13
    Width: 64, Depth: 1, Total FLOPs: 6.021e+12
    Width: 64, Depth: 2, Total FLOPs: 2.314e+13
    Width: 64, Depth: 3, Total FLOPs: 4.025e+13
    Width: 128, Depth: 1, Total FLOPs: 2.341e+13
    Width: 128, Depth: 2, Total FLOPs: 9.173e+13
    Width: 128, Depth: 3, Total FLOPs: 1.601e+14
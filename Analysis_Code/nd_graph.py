import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting


# L = A N^{-a} + B D^{-b} + L_min
def scaling_law_2d(X, A, a, B, b, L_min):
    N, D = X
    return A * (N ** (-a)) + B * (D ** (-b)) + L_min


def save_result(A, a, B, b, L_min, r2, filename):
    new_row = pd.DataFrame([{
        "Title": "NxD regression",
        "A": A,
        "a": a,
        "B": B,
        "b": b,
        "L_min": L_min,
        "r2": r2
    }])

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df["Title"] != "NxD regression"]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(filename, index=False)


def fit_nd(df, output_file):
    df = df.copy()

    # Keep clean regime
    df = df[df["depth"] == 2]

    # Average seeds
    df = df.groupby(["N", "D", "epochs"]).mean().reset_index()


    N = df["N"].values
    D = (df["D"] * df["epochs"]).values
    L = df["final_test_loss_avg_last5"].values

    # Initial guesses
    A0 = max(L)
    B0 = max(L)
    a0 = 0.15
    b0 = 0.35
    L0 = min(L) * 0.9

    params, _ = curve_fit(
        scaling_law_2d,
        (N, D),
        L,
        p0=[A0, a0, B0, b0, L0],
        bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
        maxfev=20000
    )

    A, a, B, b, L_min = params

    L_pred = scaling_law_2d((N, D), A, a, B, b, L_min)

    r2 = 1 - np.sum((L - L_pred)**2) / np.sum((L - np.mean(L))**2)

    print("NxD regression")
    print(f"R^2: {r2}")
    print(f"A, a: {A}, {a}")
    print(f"B, b: {B}, {b}")
    print(f"L_min: {L_min}")

    # ---------- 2D Sanity Plot ----------
    plt.figure()
    plt.scatter(L, L_pred, s=25)
    plt.plot([min(L), max(L)], [min(L), max(L)], linestyle="--")
    plt.xlabel("Actual Loss")
    plt.ylabel("Predicted Loss")
    plt.title("NxD Fit: Predicted vs Actual")
    plt.show()

    # ---------- 3D Surface + Points ----------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=20)

    # Scatter actual data
    ax.scatter(N, D, L, s=30, label="data")

    # Create smooth grid
    N_lin = np.linspace(min(N), max(N), 40)
    D_lin = np.linspace(min(D), max(D), 40)
    N_grid, D_grid = np.meshgrid(N_lin, D_lin)

    L_grid = scaling_law_2d((N_grid, D_grid), A, a, B, b, L_min)

    # Surface
    ax.plot_surface(N_grid, D_grid, L_grid, alpha=0.3)

    ax.set_xlabel("N (model size)")
    ax.set_ylabel("D (dataset size)")
    ax.set_zlabel("Loss")
    ax.set_title("L(N, D) Scaling Surface")

    plt.legend()
    plt.show()

    save_result(A, a, B, b, L_min, r2, filename=output_file)


if __name__ == "__main__":
    df = pd.read_csv("final_data.csv")
    fit_nd(df, output_file="nd_regression.csv")
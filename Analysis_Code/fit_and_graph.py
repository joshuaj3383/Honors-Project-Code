import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def add_csv(title, A, a, L_base, r2, filename="regression_data.csv"):
    new_row = pd.DataFrame([{
        "Title": title,
        "A": A,
        "a": a,
        "L_base": L_base,
        "r2": r2
    }])

    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = df[df["Title"] != title]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(filename, index=False)


# Model: L = A * X^{-a} + L_base
def scaling_law(x, A, a, L_base):
    return A * (x ** (-a)) + L_base


def fit_and_graph(X, Y, title, x_label, y_label, scale="normal"):
    X = np.array(X)
    Y = np.array(Y)

    # Initial guesses
    A0 = max(Y)
    a0 = 0.1
    L0 = min(Y) * 0.9

    params, _ = curve_fit(scaling_law, X, Y, p0=[A0, a0, L0], maxfev=10000)
    A, a, L_base = params

    Y_fit = scaling_law(X, A, a, L_base)

    # R^2 in original space
    r2 = 1 - np.sum((Y - Y_fit)**2) / np.sum((Y - np.mean(Y))**2)

    print(f"{title}")
    print(f"R^2: {r2}")
    print(f"A, a, L_base: {A}, {a}, {L_base}")

    plt.figure()

    X_smooth = np.linspace(min(X), max(X), 200)
    Y_smooth = scaling_law(X_smooth, A, a, L_base)

    plt.scatter(X, Y, label="data", s=25)
    plt.plot(X_smooth, Y_smooth, label="fit")

    # Scale control
    if scale == "loglog":
        plt.xscale("log")
        plt.yscale("log")
    elif scale == "semilogx":
        plt.xscale("log")
    elif scale == "semilogy":
        plt.yscale("log")
    # else: "normal" → no scaling

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

    add_csv(title, A, a, L_base, r2)


# ----------- CLEAN EXPERIMENTS ONLY -----------

def N_width(df, log=False):
    df = df[(df["depth"] == 2) & (df["D"] == 50000)]
    df = df.groupby("N").mean().reset_index()   # average seeds
    df = df.sort_values("N")

    fit_and_graph(
        df["N"].values,
        df["final_test_loss_avg_last5"].values,
        "N-Width vs Loss",
        "N",
        "Loss",
        log
    )


def N_depth(df, log=False):
    df = df[(df["width"] == 32) & (df["D"] == 50000)]
    df = df.groupby("N").mean().reset_index()
    df = df.sort_values("N")

    fit_and_graph(
        df["N"].values,
        df["final_test_loss_avg_last5"].values,
        "N-Depth vs Loss",
        "N",
        "Loss",
        log
    )


def D_general(df, log=False):
    df = df[(df["width"] == 32) & (df["depth"] == 2)]
    df = df.groupby("D").mean().reset_index()
    df = df.sort_values("D")

    fit_and_graph(
        df["D"].values,
        df["final_test_loss_avg_last5"].values,
        "D vs Loss",
        "D",
        "Loss",
        log
    )


def C_Nscaling(df, log=False):
    df = df[df["D"] == 50000]
    df = df[df["depth"] == 2]
    df = df.groupby("total_flops").mean().reset_index()
    df = df.sort_values("total_flops")

    fit_and_graph(
        df["total_flops"].values,
        df["final_test_loss_avg_last5"].values,
        "C (N-scaling) vs Loss",
        "C",
        "Loss",
        log
    )


def C_Dscaling(df, log=False):
    df = df[(df["width"] == 32) & (df["depth"] == 2)]
    df = df.groupby("total_flops").mean().reset_index()
    df = df.sort_values("total_flops")

    fit_and_graph(
        df["total_flops"].values,
        df["final_test_loss_avg_last5"].values,
        "C (D-scaling) vs Loss",
        "C",
        "Loss",
        log
    )


if __name__ == "__main__":
    df = pd.read_csv("final_data.csv")

    N_width(df, log=True)
    N_depth(df, log=True)
    D_general(df, log=True)
    C_Nscaling(df, log=True)
    C_Dscaling(df, log=True)
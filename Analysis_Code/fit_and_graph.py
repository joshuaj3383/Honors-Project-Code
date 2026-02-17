import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def add_csv(title,A, a, r2, filename="regression_data.csv"):
    new_row = pd.DataFrame([{
        "Title": title,
        "A": A,
        "a": a,
        "r2": r2
    }])

    if os.path.exists(filename):
        df = pd.read_csv(filename)

        # remove existing rows with same title
        df = df[df["Title"] != title]

        # append new row
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # overwrite file
    df.to_csv(filename, index=False)


def fit_and_graph(X, Y, title, x_label, y_label, log=False):
    # Log space
    logX = np.log(X)
    logY = np.log(Y)

    m, b = np.polyfit(logX, logY, 1)

    # L(X) = Ax^-a
    a, A = -m, np.exp(b)


    Y_pred = m * logX + b

    r2 = 1 - np.sum((logY - Y_pred)**2) / np.sum((logY - np.mean(logY))**2)

    print(f"{title}")
    print(f"R^2:")
    print(f"{r2}")
    print(f"A, a:")
    print(f"{A}, {a}")

    plt.figure()

    # General for original space
    X_fit = np.linspace(min(X), max(X), 200)
    Y_fit = A * X_fit ** (-a)

    plt.scatter(X, Y, label="data", s=25,c="blue")
    plt.plot(X_fit, Y_fit, label="fit",c="red")

    # Convert to log space if requested
    if log:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


    add_csv(title, A, a, r2)




def N_width(df,log=False):
    # Extract only varying width
    df = df[(df["depth"] == 2) & (df["D"] == 50000)]

    df = df.sort_values("N")

    N = df["N"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(N, L, "N-Width vs Loss", "N-Width", "Loss", log=log)

def N_depth(df,log=False):
    # Extract Only varying depth
    df = df[(df["width"] == 32) & (df["D"] == 50000)]

    df = df.sort_values("N") # Make sure N is increasing

    N = df["N"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(N, L, "N-Depth vs Loss", "N-Depth", "Loss", log=log)

def N_general(df,log=False):
    df = df[df["D"] == 50000]

    df = df.sort_values("N")

    N = df["N"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(N, L, "N-General vs Loss", "N-General", "Loss", log=log)


def D_general(df,log=False):
    df = df[(df["width"] == 32) & (df["depth"] == 2)]

    df = df.sort_values("D")

    D = df["D"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(D, L, "D_General vs Loss", "D_General", "Loss", log=log)


def C_Nscaling(df,log=False):
    df = df[df["D"] == 50000]

    df = df.sort_values("total_flops")

    C = df["total_flops"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(C, L, "C_Nscaling", "C_Nscaling", "Loss", log=log)

def C_Dsclaing(df,log=False):
    df = df[(df["width"] == 32)& (df["depth"] == 2)]

    df = df.sort_values("total_flops")

    C = df["total_flops"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(C, L, "C_Dscaling", "C_Dscaling", "Loss", log=log)

def C_general(df,log=False):
    df = df.sort_values("total_flops")

    C = df["total_flops"].values
    L = df["final_test_loss_avg_last5"].values

    fit_and_graph(C, L, "C_General vs Loss", "C_General", "Loss", log=log)


if __name__ == "__main__":
    file = "final_data.csv"

    with open(file) as f:
        df = pd.read_csv(f)

        N_width(df)
        N_depth(df)
        N_general(df)
        D_general(df)
        C_Nscaling(df)
        C_Dsclaing(df)
        C_general(df)

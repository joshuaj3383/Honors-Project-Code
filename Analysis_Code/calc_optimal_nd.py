import numpy as np
import pandas as pd
from Training_Code.train_model import count_parameters, proxy_compute, CNN


if __name__ == "__main__":
    width = 32
    depth = 2
    dataset_size = 50000
    epochs = 50

    C = proxy_compute(count_parameters(CNN(width,depth)), dataset_size, epochs)

    compute_data = "fixed_compute_data.csv"
    regression_data = "nd_regression.csv"

    df = pd.read_csv(compute_data)

    df.sort_values("best_test_loss",ascending=True,inplace=True)

    best = df.iloc[0]

    print(f"Tested optimal N,D tradeoff from: width={best["width"]}, epochs={best["epochs"]}")
    print("N:", best["N"])
    print("D:", best["D"]*best["epochs"])

    df = pd.read_csv(regression_data)
    # The top three all 36,36,40 so we can safely assume best is around 36

    for _, row in df.iterrows():
        A = row["A"]
        a = row["a"]
        B = row["B"]
        b = row["b"]

        k = (a*A)/(b*B)
        invAB = 1/(a+b)

        N = pow(k,invAB) * pow(C,b*invAB)
        D = pow(k, -invAB) * pow(C,a*invAB)

        print("\nOptimal N,D tradeoff from test:", row["Title"])
        print("N:", N)
        print("D:", D)




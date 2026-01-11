import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

wine_type = 'WHITE'

def compute_tolarence(y_true, y_pred, t):
    return (np.abs(y_true-y_pred) <= t).mean()

def calculate_epsilon(x_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=3)
    knn.fit(x_train, y_train)

    y_knn_pred = knn.predict(x_train)

    N = len(y_train)

    sigma_hat = (1.5 / N) * np.sum((y_train - y_knn_pred) ** 2)

    return sigma_hat / np.sqrt(N)



def main():

    if wine_type == 'WHITE':
        data = pd.read_csv(
            "data/winequality-white.csv",
            sep=";",
            header=0
        )
        gamma = 0.7
    else:
        data = pd.read_csv(
            "data/winequality-red.csv",
            sep=";",
            header=0
        )
        gamma = 0.15

    X = data.drop("quality", axis=1)
    y = data["quality"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.33,
        random_state=42
    )

    epsilon = calculate_epsilon(X_train, y_train)

    model = SVR(
        kernel="rbf",
        C=3,
        epsilon=epsilon,
        gamma=gamma
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mad = np.mean(np.abs(preds - y_test))

    print(f"MAD={mad:.4f}")
    print(compute_tolarence(y_test, preds, 1))

if __name__ == "__main__" :
    main()
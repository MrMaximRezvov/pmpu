import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    data = pd.read_csv("functions/forecast1.csv")
    print(data)
    X = np.arange(0, len(data))
    print(X)
    Y = data.iloc[:, 0]
    print(Y)
    # for i in range(len(Y)):
    #     if Y[i] == None:
    #         Y[i] == (Y[i-1] + Y[i+1]) / 2
    #     print(Y[i])
    Y[52] = 200
    model = LinearRegression()
    X_feat = np.transpose([X**2, X, np.sin(2*np.pi * 1/13 * X)])
    print(X_feat)
    model.fit(X_feat, Y)
    print(model.coef_)


    # X_pred = []
    # for i in range(12):
        # X_pred.append([len(Y) + i])
    X_pred = np.arange(len(Y), len(Y) + 12)
    Xm_pred = np.transpose([X_pred**2, X_pred, np.sin(2*np.pi * 1/13 * X_pred)])
    Y_pred = model.predict(Xm_pred)
    print(Y_pred)

    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    main()





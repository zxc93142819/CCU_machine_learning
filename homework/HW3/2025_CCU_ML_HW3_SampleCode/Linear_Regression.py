import numpy as np
import argparse
import os
from DataLoader import DataLoader
import matplotlib.pyplot as plt


import numpy as np

def Linear_Regression(DataLoader):
    """
    Perform Linear Regression.
    Returns:
    - weights: A (2x1) weight vector.
    - Ein: The in-sample error.
    """
    data = np.array([x[0] for x in DataLoader.data])
    y = np.array([x[1] for x in DataLoader.data])
    N = len(data)
    
    # Add bias term (column of ones)
    X = np.column_stack((data, np.ones(N)))
    
    # Compute (X^T X)^(-1) X^T y
    XX = X.T @ X  # X^T X
    X_inverse = np.linalg.pinv(XX)
    X_star = X_inverse @ X.T  # Pseudo-inverse computation
    weights = X_star @ y  # Optimal weights

    # Compute Ein (1/N * (pred - y)^2)
    predictions = X @ weights
    Ein = np.mean((predictions - y) ** 2)

    # compute gradient of Ein
    gradient_Ein = (2/N) * (XX @ weights - X.T @ y)

    print("Gradient of Ein:", gradient_Ein)
    return weights, Ein

def main(args):
    try:
        if args.path == None or not os.path.exists(args.path):
            raise
    except:
        print("File not found, please try again")
        exit()

    Loader = DataLoader(args.path)
    weights, Ein = Linear_Regression(DataLoader=Loader)

    # This part is for plotting the graph
    plt.title(
        'Linear Regression, Ein = %.2f' % (Ein))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    Data = np.array(Loader.data)
    plt.scatter(Data[:, 0], Data[:, 1], c='b', label='data')

    x = np.linspace(-100, 100, 10)
    # This is your regression line
    y = weights[0]*x + weights[1]
    plt.plot(x, y, 'g', label='regression line', linewidth='1')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    parse = argparse.ArgumentParser(
        description='Place the .txt file as your path input')
    parse.add_argument('--path', type=str, help='Your file path')
    args = parse.parse_args()
    main(args)

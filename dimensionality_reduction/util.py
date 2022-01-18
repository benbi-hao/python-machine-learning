import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 二维数据集决策边界可视化
def plot_decision_regions(X, y, classifier, resolution=0.02,
                          title='title', xlabel='feature1', ylabel='feature2'):
    # set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left')
    plt.show()


def wine_all():
    df_wine = pd.read_csv('../datasets/wine/wine.data')
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    return X, y


def wine_all_split():
    X, y = wine_all()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    return X_train, X_test, y_train, y_test


def wine_all_split_std():
    X_train, X_test, y_train, y_test = wine_all_split()
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test


def plot_data_pca(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                color='red', marker='^', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.show()


def plot_data_pca_flat(X, y):
    half = int(X.shape[0] / 2)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
    ax[0].scatter(X[y == 0, 0], X[y == 0, 1],
                  color='red', marker='^', alpha=0.5)
    ax[0].scatter(X[y == 1, 0], X[y == 1, 1],
                  color='blue', marker='o', alpha=0.5)
    ax[1].scatter(X[y == 0, 0], np.zeros((half, 1)) + 0.02,
                  color='red', marker='^', alpha=0.5)
    ax[1].scatter(X[y == 1, 0], np.zeros((half, 1)) - 0.02,
                  color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PC1')
    plt.show()

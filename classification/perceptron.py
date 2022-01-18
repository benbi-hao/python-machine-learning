import numpy as np
from classification.util import plot_decision_regions, plot_cost_epoch, iris_100_2


# 感知器学习
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


X, y = iris_100_2()

# 使用感知器训练并绘制每次迭代错误数量的折线图
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plot_cost_epoch(ppn.errors_, ylabel='Number of misclassifications')

# 可视化选取的两个特征和类别
plot_decision_regions(X, y, classifier=ppn,
                      title='Perceptron',
                      xlabel='sepal length [cm]',
                      ylabel='petal length [cm]')

# 随机梯度下降，每次使用一个训练样本渐进地更新权重
# 能更快收敛，但由于基于单个样本，更容易跳出小范围局部最优点
# 要让数据随机提供给算法，每次迭代要打乱数据集

# 采用随着时间变化的自适应学习速率来替代固定学习速率

# 小批次学习：例如一次使用50个样本，来进行权重更新，基于批量梯度下降和随机梯度下降之间

from numpy.random import seed
import numpy as np
from classification.util import plot_decision_regions, plot_cost_epoch, iris_100_2_std


# 自适应线性神经元-随机梯度下降
class AdalineSGD(object):
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


X_std, y = iris_100_2_std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada,
                      title='Adaline - Stochastic Gradient Descent',
                      xlabel='sepal length [standardized]',
                      ylabel='petal length [standardized]')
plot_cost_epoch(ada.cost_, ylabel='Average Cost')

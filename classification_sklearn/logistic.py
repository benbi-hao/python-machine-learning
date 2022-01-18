import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from classification_sklearn.util import plot_decision_regions, iris_all_2_split_std, combine
# 读取数据
X_train_std, X_test_std, y_train, y_test = iris_all_2_split_std()
# logistic回归分类，与adaline相比，将激励函数改成了sigmoid函数，而adaline的激励函数是恒等函数
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
X_combined_std, y_combined = combine(X_train_std, X_test_std, y_train, y_test)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')
# 对于同一个分类任务，使用不同的正则化系数，观察权重系数的变化
weights, params = [], []
for c in range(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

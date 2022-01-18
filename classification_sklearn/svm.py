import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from classification_sklearn.util import plot_decision_regions, iris_all_2_split_std, combine

# 读取数据
X_train_std, X_test_std, y_train, y_test = iris_all_2_split_std()
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
X_combined_std, y_combined = combine(X_train_std, X_test_std, y_train, y_test)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')

# # sklearn创建可在线学习(partial_fit)的分类器
# from sklearn.linear_model import SGDClassifier
# ppn = SGDClassifier(loss='perceptron')
# lr = SGDClassifier(loss='log')
# svm = SGDClassifier(loss='hinge')

# 核svm用于解决非线性可分问题
# 创建一个非线性可分异或数据集
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='x', label='l')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='r', marker='s', label='-l')
plt.ylim(-3.0)
plt.legend()
plt.show()

# 使用核函数对异或数据集分类
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)

# gamma值，是rbf核函数中的参数，小的话就宽松，大的话就紧凑
# gamma=0.2
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')
# gamma=100.0
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')

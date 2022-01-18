from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from classification_sklearn.util import plot_decision_regions, iris_all_2_split_std, combine

# 读取数据
X_train_std, X_test_std, y_train, y_test = iris_all_2_split_std()
# 训练感知器模型
ppn = Perceptron(n_iter_no_change=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)
# 预测
y_pred = ppn.predict(X_test_std)
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 准确率
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
# 绘制决策区域
X_combined_std, y_combined = combine(X_train_std, X_test_std, y_train, y_test)
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')

# k-近邻算法
# 是一个非参数化模型，不用先设定参数，对样本进行训练
# 基本思想是找到与待预测特征最近的k个样本，进行估计
from sklearn.neighbors import KNeighborsClassifier
from classification_sklearn.util import plot_decision_regions, iris_all_2_split_std, combine
# 读取数据
X_train_std, X_test_std, y_train, y_test = iris_all_2_split_std()
knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
knn.fit(X_train_std, y_train)
X_combined_std, y_combined = combine(X_train_std, X_test_std, y_train, y_test)
plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150),
                      xlabel='petal length [standardized]',
                      ylabel='petal width [standardized]')

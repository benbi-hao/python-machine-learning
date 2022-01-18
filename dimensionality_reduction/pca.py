# 主成分分析 principal component analysis，是以优化方差为目的的无监督数据降维技术
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from dimensionality_reduction.util import plot_decision_regions, wine_all_split_std
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# 读取数据并标准化
X_train_std, X_test_std, y_train, y_test = wine_all_split_std()

# 使用numpy.cov得到协方差矩阵
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenValues \n%s' % eigen_vals)

# 绘制特征值方差贡献率
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in
           sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

# 选取头两个特征值与特征向量构造子空间
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

# 转换后的特征
X_train_pca = X_train_std.dot(w)

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0],
                X_train_pca[y_train == l, 1],
                c=c, label=l, marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# 使用sklearn进行主成分分析
pca = PCA(n_components=2)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier=lr,
                      xlabel='PC1', ylabel='PC2')

# 查看在测试集上的效果
plot_decision_regions(X_test_pca, y_test, classifier=lr,
                      xlabel='PC1', ylabel='PC2')

# 保留所有主成分
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)
print('方差贡献率:', pca.explained_variance_ratio_)

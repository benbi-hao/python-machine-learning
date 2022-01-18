import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numpy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA, KernelPCA
# from matplotlib.ticker import FormatStrFormatter
from dimensionality_reduction.util import plot_data_pca, plot_data_pca_flat


def rbf_kernel_pca(X, gamma, n_components):
    # 计算样本间距离
    sq_dists = pdist(X, 'sqeuclidean')
    # 转化距离为距离矩阵
    mat_sq_dists = squareform(sq_dists)
    # 计算对称核矩阵
    K = exp(-gamma * mat_sq_dists)
    # 中心化核矩阵
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 获取中心化后核矩阵的特征值与特征向量
    eigvals, eigvecs = eigh(K)
    # 选取前k个特征向量
    X_pc = np.column_stack((eigvecs[:, -i]
                           for i in range(1, n_components + 1)))
    return X_pc


# 使用半月形数据
X, y = make_moons(n_samples=100, random_state=123)
plot_data_pca(X, y)

# 使用scikit的PCA分离半月形数据
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
plot_data_pca_flat(X_spca, y)

# 使用自定义核PCA函数
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
plot_data_pca_flat(X_kpca, y)
# ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
# ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

# 使用同心圆数据
X, y = make_circles(n_samples=1000,
                    random_state=123, noise=0.1, factor=0.2)
plot_data_pca(X, y)

# 使用scikit的PCA分离同心圆数据
scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
plot_data_pca_flat(X_spca, y)

# 给定合适的gamma值，使用自定义的gamma函数
X_kpca = rbf_kernel_pca(X, gamma=15, n_components=2)
plot_data_pca_flat(X_kpca, y)


# 返回值为特征向量与特征值的rbf_kernel_pca
def rbf_kernel_pca_eig(X, gamma, n_components):
    # 计算样本间距离
    sq_dists = pdist(X, 'sqeuclidean')
    # 转化距离为距离矩阵
    mat_sq_dists = squareform(sq_dists)
    # 计算对称核矩阵
    K = exp(-gamma * mat_sq_dists)
    # 中心化核矩阵
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    # 获取中心化后核矩阵的特征值与特征向量
    eigvals, eigvecs = eigh(K)
    # 选取前k个特征向量
    alphas = np.column_stack((eigvecs[:, -i]
                           for i in range(1, n_components + 1)))
    # 选取对应的特征值
    lambdas = [eigvals[-i] for i in range(1, n_components + 1)]
    return alphas, lambdas


# 创建一个新的半月形数据集
X, y = make_moons(n_samples=100, random_state=123)
alphas, lambdas = rbf_kernel_pca_eig(X, gamma=15, n_components=1)
x_new = X[25]
x_proj = alphas[25]


def project_x(x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array([np.sum(
        (x_new - row) ** 2) for row in X])
    k = exp(-gamma * pair_dist)
    return k.dot(alphas / lambdas)


# 重现新样本的映射
x_reproj = project_x(x_new, X, gamma=15, alphas=alphas, lambdas=lambdas)
plt.scatter(alphas[y == 0, 0], np.zeros((50,)),
            color='red', marker='^', alpha=0.5)
plt.scatter(alphas[y == 1, 0], np.zeros((50,)),
            color='blue', marker='o', alpha=0.5)
plt.scatter(x_proj, 0, color='black',
            label='original projection of point X[25]',
            marker='^', s=100)
plt.scatter(x_reproj, 0, color='green',
            label='remapped point X[25]',
            marker='x', s=500)
plt.legend(scatterpoints=1)
plt.show()

# scikit中的核PCA
X, y = make_moons(n_samples=100, random_state=123)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from classification_sklearn.util import plot_decision_regions, iris_all_2_split_std, combine
# def sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))
# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
# plt.axhline(y=0.5, ls='dotted', color='k')
# plt.yticks([0.0, 0.5, 1.0])
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')
# plt.show()
#
# iris = datasets.load_iris()
# X = iris.data[:, [2, 3]]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=0
# )
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
# X_combined_std = np.vstack((X_train_std, X_test_std))
# y_combined = np.hstack((y_train, y_test))
# lr = LogisticRegression(C=1000.0, random_state=0)
# lr.fit(X_train_std, y_train)
# plot_decision_regions(X_combined_std, y_combined,
#                       classifier=lr, test_idx=range(105, 150),
#                       xlabel='petal length [standardized]',
#                       ylabel='petal width [standardized]')
# print(lr.predict_proba(X_test_std[:2, :]))
#
# weights, params = [], []
# for c in range(-5, 5):
#     lr = LogisticRegression(C=10**c, random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1])
#     params.append(10**c)
# weights = np.array(weights)
# plt.plot(params, weights[:, 0],
#          label='petal length')
# plt.plot(params, weights[:, 1], linestyle='--',
#          label='petal width')
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# plt.show()

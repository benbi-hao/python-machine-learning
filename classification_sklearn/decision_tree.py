# 树节电不纯度衡量标准比较，不纯度越低，信息增量越大
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from classification_sklearn.util import iris_all_2_split, combine, plot_decision_regions
# 可视化不纯度衡量标准区别


def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def error(p):
    return 1 - np.max([p, 1 - p])


def plot_impurity_measure():
    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    sc_ent = [e*0.5 if e else None for e in ent]
    err = [error(i) for i in x]
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                              ['Entropy', 'Entropy (scaled)',
                               'Gini Impurity', 'Misclassification Error'],
                              ['-', '-', '--', '-.'],
                              ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab,
                       linestyle=ls, lw=2, color=c)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
    plt.show()


# 对比不同不纯度衡量方法
plot_impurity_measure()

# 读取数据
X_train, X_test, y_train, y_test = iris_all_2_split()
# 构建决策树
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)
X_combined, y_combined = combine(X_train, X_test, y_train, y_test)
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150),
                      xlabel='petal length [cm]',
                      ylabel='petal width [cm]')

# 随机森林，基于决策树，训练多棵决策树，再进行多数投票
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,
                                random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150),
                      xlabel='petal length',
                      ylabel='petal width')

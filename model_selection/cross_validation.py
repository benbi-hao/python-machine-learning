# holdout将数据集划分为训练集验证集和测试集，对数据集划分方法敏感
# k-fold对训练数据集进行k次不同的划分，得到k个性能评估，取平均值，因此对数据集划分方法不敏感
# k标准值为10，如果数据集更小，则需要加大k值
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from model_selection.util import wdbc_all_split
X_train, X_test, y_train, y_test = wdbc_all_split()

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])
# pipe_lr.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
scores = []
for k, (train, test) in enumerate(kfold.split(X_train, y_train)):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f'
          % (k + 1, np.bincount(y_train[train]), score))
print('CV accuracy: %.3f +/- %.3f'
      % (np.mean(scores), np.std(scores)))

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f'
      % (np.mean(scores), np.std(scores)))


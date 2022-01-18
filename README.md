# Python机器学习（基于python3.8）

## 简介

本仓库代码来源于《Python机器学习》（机械工业出版社）（原书为《Python Machine Learning》<sup>[1]</sup>），也就是本人阅读书籍时，从书中摘下组合的代码。代码做了简单注释并且可以运行。

书中部分实现是使用python手动实现，还有部分是使用sklearn实现。

## Python版本

Python 3.8.6

## 仓库结构

- [1_简单分类器](./classification)
  - [感知器](./classification/perceptron.py)
  - [自适应线形神经元-批量梯度下降](./classification/adalineGD.py)
  - [自适应线形神经元-随机梯度下降](./classification/adalineSGD.py)
  - [重用代码*](./classification/util.py)
- [2_sklearn实现的分类器](./classification_sklearn)
  - [感知器](./classification_sklearn/perceptron.py)
  - [逻辑回归](./classification_sklearn/logistic.py)
  - [支持向量机](./classification_sklearn/svm.py)
  - [决策树](./classification_sklearn/decision_tree.py)
  - [K-近邻](./classification_sklearn/knn.py)
  - [重用代码*](./classification_sklearn/util.py)
- [3_预处理](./preprocessing)
  - [sklearn的预处理](./preprocessing/preprocessing.ipynb)
- [4_降维](./dimensionality_reduction)
  - [主成分分析](./dimensionality_reduction/pca.py)
  - [核主成分分析](./dimensionality_reduction/kernal_pca.py)
  - [线形判别分析](./dimensionality_reduction/lda.py)
  - [重用代码*](./dimensionality_reduction/util.py)
- [5_模型评估和参数调优](./model_selection)
  - [sklearn实现的管道](./model_selection/pipeline.py)
  - [k折交叉验证](./model_selection/cross_validation.py)
  - [学习和验证曲线](./model_selection/learning_curve.py)
  - [网格搜索](./model_selection/grid_search.py)
  - [性能评估指标](./model_selection/evaluation_index.py)
  - [重用代码*](./model_selection/util.py)
- [6_集成学习](./ensemble)
  - [多数投票](./ensemble/majority_vote.py)
  - [套袋](./ensemble/bagging.py)
  - [自适应增强](./ensemble/ada_boost.py)
  - [重用代码*](./ensemble/util.py)
- [7_情感分析](./sentiment)
  - [词袋模型](./sentiment/bag_of_words.py)
  - [对IMDb数据集分析](./sentiment/imdb.py)
  - [重用代码](./sentiment/util.py)
- [8_回归分析](./regression)
  - [线形回归](./regression/lin_reg.py)
  - [线形回归评估方法](./regression/lin_reg_assessment.py)
  - [多项式回归](./regression/poly_reg.py)
  - [用正则化方法回归](./regression/regularization.py)
  - [重用代码*](./regression/util.py)
- [9_聚类分析](./clustering)
  - [k-均值聚类](./clustering/k-means.py)
- [10_神经网络](./mlp)
  - [神经网络简单实现](./mlp/neuralnet.py)
  - [重用代码*](./mlp/util.py)
- [数据集](./datasets)
  - [aclImdb](./datasets/aclImdb/README.md)
  - [housing](./datasets/housing/README.md)
  - [iris](./datasets/iris/README.md)
  - [mnist](./datasets/mnist/README.md)
  - [wdbc](./datasets/wdbc/README.md)
  - [wine](./datasets/wine/README.md)
- [Python环境要求](./requirements.txt)
- [README.md](./README.md)

## 参考文献

[1] Raschka S. Python machine learning[M]. Packt publishing ltd, 2015.
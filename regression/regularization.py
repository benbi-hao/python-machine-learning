# 岭回归
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
# LASSO回归
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0)
# ElasticNet
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)

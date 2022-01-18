import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def wdbc_all():
    df = pd.read_csv('../datasets/wdbc/wdbc.data', header=None)
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    return X, y


def wdbc_all_split():
    X, y = wdbc_all()
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.20, random_state=1)
    return X_train, X_test, y_train, y_test

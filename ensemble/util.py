import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def wine_all():
    # 读取葡萄酒数据集
    df_wine = pd.read_csv('../datasets/wine/wine.data', header=None)
    df_wine.columns = [
        'Class label', 'Alcohol',
        'Malic acid', 'Ash',
        'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity', 'Hue',
        'OD280/OD315 of diluted wines',
        'Proline'
    ]
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'Hue']].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


def wine_all_split():
    X, y = wine_all()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=1
    )
    return X_train, X_test, y_train, y_test

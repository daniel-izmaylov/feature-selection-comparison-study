import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
import numpy as np
import scipy.io
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder


def read_bioconductor(db):
    df = pd.read_csv(db)
    df = df.T
    new_header = df.iloc[0]
    df = df[1:]
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0]
    y = fillna(y)

    return X, y

def read_scikit_mat(db):
    mat = scipy.io.loadmat(db)
    X = mat['X']
    y = mat['Y'][:, 0]

    return X, y

def read_ARFF(db):
    df = arff.loadarff(db)
    df = pd.DataFrame(df[0])
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1]
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)
    return X, y

def read_datamicroarray(db):
    df = pd.read_csv(db, header=None)

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1]
    y = fillna(y)

    return X, y

    return

def set_categories(X):
    for col in X:
        if X[col].dtype.name == 'object':
            X[col] = X[col].astype(float)
    return X

def imputation(X):
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imp_mean.fit_transform(X)
    return X

def variance_normalization(X):
    selector = VarianceThreshold()
    X = selector.fit_transform(X)

    pt = PowerTransformer()
    X = pt.fit_transform(X)
    return X

def fillna(y):
    y = y.astype(str).astype(float)
    y = y.fillna(y.max() + 1)
    y = y.astype(int).values
    return y

def read_and_fix(db):
    function_name = {'bioconductor': read_bioconductor, 'scikit': read_scikit_mat, 'ARFF': read_ARFF, 'datamicroarray': read_datamicroarray}

    file_type = db.split("/")[1].split("_")[0]

    X, y = function_name[file_type](db)
    # X = set_categories(X)
    # TODO: Make sure that the imputation works correctly (because categories might be wrong):
    X = imputation(X)
    X = variance_normalization(X)

    return X, y

#TODO: The y values are wrong for datamicroarray. 1) Fix it. 2) Make sure it's correct for other types
read_and_fix('Data/datamicroarray_chiaretti.csv')



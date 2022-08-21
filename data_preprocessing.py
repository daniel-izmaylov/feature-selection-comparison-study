import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
import numpy as np
import scipy.io
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os

class imputation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imp_mean.fit_transform(X)
        return X

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        X = self.fit(X)
        return X

class variance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        selector = VarianceThreshold()
        X = selector.fit_transform(X)

        pt = PowerTransformer()
        self.X = pt.fit_transform(X)
        return X

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        X = self.fit(X)
        return X

def fillna(X, y):
    y = y.astype(str).astype(float)
    y = y.fillna(y.max() + 1)
    y = y.astype(int).values
    return X, y

def read_bioconductor(db):
    df = pd.read_csv(db, header=0)
    df = df.T
    cols = df.iloc[0].values
    df = df[1:]
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0]
    X, y = fillna(X, y)
    return X, y, cols

def read_scikit_mat(db):
    mat = scipy.io.loadmat(db)
    X = mat['X']
    y = mat['Y'][:, 0]
    cols = np.arange(X.shape[1] + 1)
    return X, y, cols

def read_ARFF(db):
    df = arff.loadarff(db)
    df = pd.DataFrame(df[0])
    cols = df.columns
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1]
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)
    return X, y, cols

def read_datamicroarray(db):
    df = pd.read_csv(db, header=None)
    X = df.iloc[:, :-1]
    cols = np.arange(X.shape[1] + 1)
    X = X.to_numpy()
    y = df.iloc[:, -1]
    X, y = fillna(X, y)
    return X, y, cols

def read_and_fix(db):
    function_name = {'bioconductor': read_bioconductor, 'scikit': read_scikit_mat, 'ARFF': read_ARFF, 'datamicroarray': read_datamicroarray}
    file_type = db.split("/")[1].split("_")[0]
    X, y, cols = function_name[file_type](db)
    return X, y, cols

def to_csv(X, y, cols, name):
    suffix_name = name.split("/")[1].split(".")[0]
    df_array = np.column_stack((X,y))
    df = pd.DataFrame(df_array)
    df.to_csv('after_preprocess/' + suffix_name + ".csv", index=False, header=cols)
    
path = 'Data/'
all_files = []
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file)):
        all_files.append(os.path.join(path,file))

for name in all_files:
    X, y, cols = read_and_fix(name)
    pipe = Pipeline(steps=[('imputation', imputation()), ('normalization', variance())])
    pipe.fit(X, y)
    X = pipe.transform(X)
    to_csv(X, y, cols, name)


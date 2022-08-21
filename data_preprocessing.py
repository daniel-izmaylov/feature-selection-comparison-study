import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
import numpy as np
import scipy.io
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import os

class imputation(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.X = X
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imp_mean.fit_transform(X)
        return X

    def transform(self, X):
        return self.X

    def fit_transform(self, X, y=None):
        X = self.fit(X)
        return X

    def get_feature_names_out(self, features=None):
        return self.X.columns

class variance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        #selector = VarianceThreshold()
        #X = selector.fit_transform(X)

        pt = PowerTransformer()
        X = pt.fit_transform(X)
        return X

    def transform(self, X):
        return self.X

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
    cols = df.iloc[0, :]
    cols = cols[1:]
    df = df[1:]
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X, y = fillna(X, y)
    X.columns = cols
    return X, y

def read_scikit_mat(db):
    mat = scipy.io.loadmat(db)
    X = mat['X']
    X = pd.DataFrame(X)
    y = mat['Y'][:, 0]
    return X, y

def read_ARFF(db):
    df = arff.loadarff(db)
    df = pd.DataFrame(df[0])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    label_encoder = LabelEncoder().fit(y)
    y = label_encoder.transform(y)
    return X, y

def read_datamicroarray(db):
    df = pd.read_csv(db, header=None)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X, y = fillna(X, y)
    return X, y

def read_and_fix(db):
    function_name = {'bioconductor': read_bioconductor, 'scikit': read_scikit_mat, 'ARFF': read_ARFF, 'datamicroarray': read_datamicroarray}
    file_type = db.split("/")[1].split("_")[0]
    X, y = function_name[file_type](db)
    return X, y

def to_csv(X, y, name, cols):
    suffix_name = name.split("/")[1].split(".")[0]
    df_array = np.column_stack((X,y))
    df = pd.DataFrame(df_array)
    cols = np.append(cols, "label")
    df.to_csv('after_preprocess/' + suffix_name + ".csv", index=False, header=cols)
    
path = 'Data/'
all_files = []
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file)):
        all_files.append(os.path.join(path,file))

for name in all_files:
    X, y = read_and_fix(name)
    cols = X.columns
    pipe = Pipeline(steps=[('imputation', imputation()), ('variance_thresh', VarianceThreshold()), ('normalization', StandardScaler()),
                           ('power_transformer', PowerTransformer())])
    X = pipe.fit_transform(X, y)
    cols = pipe.get_feature_names_out(cols)
    to_csv(X, y, name, cols)


# %%
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# pd.options.plotting.backend = "plotly"

# %%
import scipy.io
mat =scipy.io.loadmat('/home/izmaylov/test/feature_selection/scikit-Dataset/lymphoma.mat')
X=mat['X']
y = mat['Y'][:, 0] 



# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# %%
from skfeature.function.information_theoretical_based import MRMR
# num_fea = 5  
# idx  = MRMR.mrmr(X_train, y_train)
# selected_features_train = X_train[:, idx[0:num_fea]]
# selected_features_test = X_test[:, idx[0:num_fea]]

# %%
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe

# %%
from sklearn.model_selection import GridSearchCV
#import logistic_regression 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#import naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFdr

from tempfile import mkdtemp
from joblib import Memory
from shutil import rmtree

# search_space = [{'selector__k': [5, 10, 20, 30]},
#                 {'classifier': [LogisticRegression(solver='lbfgs')]},
#                 {'classifier': [RandomForestClassifier(n_estimators=100)]}]
#                 # {'classifier': [KNeighborsClassifier()]},
#                 # {'classifier': [GaussianNB()]},
#                 # {'classifier': [SVC()]}]
# # grid_search = GridSearchCV(pipe, param_grid=search_space)
cachedir = mkdtemp()
location = "cachedir"
memory = Memory(location=location, verbose=False)

# pipe = Pipeline([('selector', SelectKBest(mutual_info_classif, k=5)),
#                  ('classifier', LogisticRegression())],memory=memory)

# %%
import itertools
def make_param_grids(steps, param_grids):

    final_params=[]

    # Itertools.product will do a permutation such that 
    # (pca OR svd) AND (svm OR rf) will become ->
    # (pca, svm) , (pca, rf) , (svd, svm) , (svd, rf)
    for estimator_names in itertools.product(*steps.values()):
        current_grid = {}

        # Step_name and estimator_name should correspond
        # i.e preprocessor must be from pca and select.
        for step_name, estimator_name in zip(steps.keys(), estimator_names):
            for param, value in param_grids.get(estimator_name).items():
                if param == 'object':
                    # Set actual estimator in pipeline
                    current_grid[step_name]=[value]
                else:
                    # Set parameters corresponding to above estimator
                    current_grid[step_name+'__'+param]=value
        #Append this dictionary to final params            
        final_params.append(current_grid)

    return final_params

# %%


# %%
from sklearn.feature_selection import SelectFdr, chi2, f_classif
SelectFdr(f_classif, alpha=0.1)

"""example of param_grid:
https://stackoverflow.com/questions/42266737/parallel-pipeline-to-get-best-model-using-gridsearch

pipeline_steps = {'preprocessor':['pca', 'select'],
                  'classifier':['svm', 'rf']}
# fill parameters to be searched in this dict
all_param_grids = {'svm':{'object':SVC(),
                         }, 

                   'rf':{'object':RandomForestClassifier(),
                        },

                   'pca':{'object':PCA(),
                          'n_components':[10,20]
                         },

                   'select':{'object':SelectKBest(),
                             'k':[5,10]
                            }
                  }  

{
      "SelectFdr":{"object":{SelectFdr(f_classif, alpha=0.1)}},
      # "MRMR":{"object":{MRMR.mrmr()}},
      'pca':{'object':PCA()},
      # "LogisticRegression": {"object": LogisticRegression(solver='lbfgs')},
      # "RandomForest": {"object": RandomForestClassifier(n_estimators=100)},
      # "KNeighbors": {"object": KNeighborsClassifier()},
      "naive_bayes": {"object": GaussianNB()},
      # "SVD":      {"object": SVC()}
}
"""
from sklearn.base import BaseEstimator, TransformerMixin 
# import  sklearn.metrics._scorer:
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

# class Test(BaseEstimator, TransformerMixin ):
#       def __init__(self, k=5):
#             self.k = k

#       def fit(self, X, y=None):
#             self.idx  = MRMR.mrmr(X_train, y_train)
#             return self

#       def fit_transform(self, X, y=None, **fit_params):
#         if y is None:
#             # fit method of arity 1 (unsupervised transformation)
#             return self.fit(X, **fit_params).transform(X)
#         else:
#             # fit method of arity 2 (supervised transformation)
#             return self.fit(X, y, **fit_params).transform(X)
            
#       def transform(self, X):
#             return X[:, self.idx[0:self.k]]

# def Test(*args):
#     print(args)
#     return args
# pipeline_steps = {'preprocessor':['pca', 'SelectFdr',"Mrmr"],
pipeline_steps = {'preprocessor':["MRMR","SelectFdr"],
                  'classifier':['naive_bayes',"KNeighbors"]}
# X_new = SelectKBest(score_func= MRMR.mrmr, k=20).fit_transform(X, y)
def on_step(optim_result):
      # print(optim_result)
      return True
#     score = searchcv.best_score_
#     print("best score: %s" % score)
#     if score >= 0.98:
#         print('Interrupting!')
#         return True

all_param_grids =   {
                  "SelectFdr":{"object":SelectFdr(),
                              "score_func":[f_classif],
                              "alpha":[0.1]},
                  'pca':{'object':PCA()},
                  # 'MRMR':{'object':SelectKBest(MRMR.mrmr),
                  'MRMR':{'object':SelectKBest(),
                             'k':[5],
                            },
                  "naive_bayes": {"object": GaussianNB()},
                  "KNeighbors": {"object": KNeighborsClassifier()},

            }
# t=Test()
# t.fit(X_train, y_train)
# t.transform(X_train)
# Call the method on the above declared variables
param_grids_list = make_param_grids(pipeline_steps, all_param_grids)
cachedir = mkdtemp()
location = "cachedir"
memory = Memory(location=location, verbose=False)

pipe = Pipeline(steps=[('preprocessor',PCA()), ('classifier', SVC())],memory=memory)  
scoring = {
      # "roc_auc": make_scorer(roc_auc_score),
            "accuracy": make_scorer(accuracy_score),
            # "matthews_corrcoef": make_scorer(matthews_corrcoef),
            # "PR-AUC": make_scorer(average_precision_score),
            }

print(param_grids_list)


grd = GridSearchCV(pipe, param_grid = param_grids_list,verbose=False, scoring=scoring,refit=False)
grd.fit(X, y,callback=[on_step])
print("f")

# %%
estimator

# %%
memory.eval("grd.best_estimator_")

# %%
#gridseach result to dataframe
df = pd.DataFrame(grd.cv_results_)
df

# %%
grd.best_estimator_

# %%
grd.estimator.named_steps['preprocessor'].get_params().keys()

# %%
fs = grd.best_estimator_.named_steps['fs']


# %%
pipe[:-1].get_feature_names_out()


# %%


# %%
x.classifier

# %%
pipe.select

# %%
pipe.steps[0]

# %%
pipe.get_params()["preprocessor"]clf.steps[0]

# %%
X

# %%


# %%
memory.clear(warn=False)
rmtree(location)



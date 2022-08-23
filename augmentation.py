import csv

from imblearn import over_sampling
from imblearn.over_sampling import RandomOverSampler 
from sklearn.decomposition import KernelPCA

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from os import listdir
from os.path import isfile, join
from feature_algo import dssa, New_dssa
from sklearn.feature_selection import SelectKBest, RFE
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import SelectFdr, f_classif
import ReliefF

from comparison import get_fold, evalute
from timeit import default_timer as timer
import utlis
from comparison import feature_selection
import warnings
import random
 
warnings.filterwarnings("ignore")
def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

CLASSIFIERS = {'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True)}
# FS_ALGORITHMS = {'dssa': SelectKBest(dssa.fit,k=100), 'New_dssa': SelectKBest(New_dssa.fit,k=100), 'f_classif': SelectFdr(score_func=f_classif,alpha=0.1),
#                  'MRMR': SelectKBest(score_func=MRMR.mrmr,k=100), 'ReliefF': SelectKBest(score_func=ReliefF.ReliefF,k=100),
#                  'SVM': RFE(SVC(kernel='linear'),n_features_to_select=100,step=1)}

def get_best_config(file_name):
    head = ['index', 'value', 'infrence_time', 'Learning algorithm', 'Number of features selected (K)', 'chosen_features', 'Selected Features scores',
            'Filtering Algorithm', 'Fold', 'fs_time', 'Dataset Name', 'Number of samples', 'Original Number of features', 'CV Method']

    # with open("augmentation_t.csv", "a", newline="") as fn:
    #     write = csv.writer(fn)
    #     write.writerow(head)

    # path = 'results/'
    # all_files = []
    # for file in os.listdir(path):
    #     if os.path.isfile(os.path.join(path, file)):
    #         all_files.append(os.path.join(path, file))

    # for i,name in enumerate(all_files):

    print("Processing file: " + " " + file_name)
    # db=pd.read_csv("after_preprocess/"+file_name+".csv",header=0)

    df = pd.read_csv("results/"+file_name, header=0)
    best_results = return_best_config(df)
    feature_num = best_results['Number of features selected (K)']
    classifier = best_results['Learning algorithm']
    fs_algorithm = best_results['Filtering Algorithm']
    evaluate_augmentation(file_name, feature_num, classifier, fs_algorithm)

def return_best_config(df):
    t = df[df["index"]=="AUC"].groupby(["Filtering Algorithm","Number of features selected (K)","Learning algorithm"]).mean().reset_index()
    t = t.sort_values(by=["value"],ascending=False).iloc[0]
    return t

def evaluate_augmentation(file_name, feature_num, classifier, fs_algorithm):
    db_name = file_name.split(".")[0].replace("_results", "")
    df = pd.read_csv("after_preprocess/" + db_name + ".csv", header=0)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    fold_func = get_fold(X)

    y_prob_all = []
    y_pred_all = []
    y_test_all = []
    cls_time_all = []
    i = 1

    for train, test in fold_func.split(X, y):
        print(i)
        xtrain = X.iloc[train,:]
        xtest = X.iloc[test,:]
        ytrain = y[train]
        ytest = y[test]
        xtrain, xtest, ytrain, ytest, k_best_scores, k_best_index, fs_time = create_pca(feature_num, fs_algorithm, xtrain, xtest, ytrain, ytest)
        colums= list(xtrain.columns)
        # print(colums)
        xtrain, ytrain = make_augmentation(xtrain, ytrain)

        ypred, cls_time = run_cls(classifier, xtrain, ytrain, xtest, ytest)

        if (X.shape[0] <= 100):
            y_prob_all.append(ypred[:,1])
            y_pred_all.append(np.argmax(ypred, axis=1))
            cls_time_all.append(cls_time)
            y_test_all.append(ytest)
        else:
            res = evalute(ytest, np.argmax(ypred, axis=1), ypred)
            prep_results(res, cls_time/len(xtrain), classifier, feature_num, colums, k_best_scores, "Aug_" + fs_algorithm,
                         i, fs_time, db_name, X.shape[0], X.shape[1], utlis.get_fold(X.shape[0]),db_name)

        i += 1

    if (X.shape[0] <= 100):
        y_test_all = np.array(y_test_all)
        y_prob_all = np.array(y_prob_all)
        res = evalute(y_test_all,y_pred_all,y_prob_all,loo=True)
        prep_results(res, np.mean(cls_time_all) / len(xtrain), classifier, feature_num, colums, k_best_scores, "Aug_" + fs_algorithm,
                     0, fs_time, db_name, X.shape[0], X.shape[1], utlis.get_fold(X.shape[0]),db_name)

    return

def run_cls(classifier, xtrain, ytrain, xtest, ytest):
    start = timer()
    cf = CLASSIFIERS[classifier].fit(xtrain, ytrain)
    ypred = cf.predict_proba(xtest)
    cls_time = timer() - start
    return ypred, cls_time

def prep_results(res, cls_time, cls, k, chosen, chosen_scores, fs, fold, fs_time, df_name, size, all_features, cv_method,db_name):
    rows = []
    for key, value in res.items():
        rows.append([key, value, cls_time, cls, k, chosen, chosen_scores, fs, fold, fs_time, df_name, size, all_features, cv_method])

    with open("augmenrataion_res/"+db_name+"augmentation.csv", "a", newline="") as fn:
        write = csv.writer(fn)
        write.writerows(rows)



def create_pca(feature_num, fs_algorithm, xtrain, xtest, ytrain, ytest):
    start = timer()
    selected,time = feature_selection(fs_algorithm,xtrain.values, ytrain.values)
    total = timer() - start
    if fs_algorithm=="ReliefF":
        k_best_index = np.argpartition(selected, -feature_num)[-feature_num:]
        k_best_scores=selected

    elif fs_algorithm=="SVD" or fs_algorithm=="RFE" or fs_algorithm=="SVC":
        k_best_index=selected[-feature_num:]
        k_best_scores=selected[-feature_num:]
    else:
        k_best_index = np.argpartition(selected.scores_, -feature_num)[-feature_num:]
        k_best_scores=selected.scores_
    # k_best_index = np.argpartition(selected.scores_, -feature_num)[-feature_num:]
    xtrain = xtrain.iloc[:, k_best_index].reset_index(drop=True)
    xtest = xtest.iloc[:, k_best_index].reset_index(drop=True)

    linear_pca = KernelPCA(kernel='linear')
    linear_pca_features = linear_pca.fit_transform(xtrain, ytrain)
    linear_pca_features_test = linear_pca.transform(xtest)
    rbf_pca = KernelPCA(kernel='rbf')
    rbf_pca_features = rbf_pca.fit_transform(xtrain, ytrain)
    rbf_pca_features_test = rbf_pca.transform(xtest)

    xtrain = xtrain.join(pd.DataFrame(linear_pca_features), rsuffix="_linear")
    xtrain = xtrain.join(pd.DataFrame(rbf_pca_features), rsuffix="_rbf")
    xtest = xtest.join(pd.DataFrame(linear_pca_features_test), rsuffix="linear")
    xtest = xtest.join(pd.DataFrame(rbf_pca_features_test), rsuffix="_rbf")

    return xtrain, xtest, ytrain, ytest, k_best_scores, k_best_index, total

def make_augmentation(xtrain, ytrain):
    rm = RandomOverSampler(random_state=42)
    sm = over_sampling.SMOTE(k_neighbors=2,random_state=101)
    X_res, Y_res= rm.fit_resample(xtrain.values, ytrain)
    X_res, Y_res = sm.fit_resample(X_res, Y_res)
    return  X_res, Y_res

if __name__ == "__main__":
    try:
        task_n= int(os.environ['SLURM_ARRAY_TASK_ID'])
    except KeyError:
        task_n=11
    freeze_seed(55)

    files= [f for f in listdir("results") if isfile(join("results", f))]
    files.sort()
    file_name=files[task_n]
    get_best_config(file_name)


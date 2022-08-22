from imblearn import over_sampling
from sklearn.decomposition import KernelPCA

from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from feature_algo import dssa, New_dssa
from sklearn.feature_selection import SelectKBest
from skfeature.function.information_theoretical_based import MRMR
from sklearn.feature_selection import SelectFdr, f_classif
import ReliefF

CLASSIFIERS = {'LogisticRegression': LogisticRegression(),
            # 'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True)}
FS_ALGORITHMS = {'dssa': dssa, 'New_dssa': New_dssa, 'f_classif': SelectFdr(score_func=f_classif,alpha=0.1),
                 'MRMR': SelectKBest(score_func=MRMR.mrmr,k=100), 'ReliefF': SelectKBest(score_func=ReliefF.ReliefF,k=100)}

def get_best_config():
    path = 'results/'
    all_files = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            all_files.append(os.path.join(path, file))

    for name in all_files:
        best_results = pd.read_csv(name, header=0).iloc[0]
        feature_num = best_results['Number of features selected (K)']
        classifier = best_results['Learning algorithm']
        fs_algorithm = best_results['Filtering Algorithm']
        db_name = name.split("/")[1].split(".")[0].replace("_results", "")
        evaluate_augmentation(db_name, feature_num, classifier, fs_algorithm)


def evaluate_augmentation(db_name, feature_num, classifier, fs_algorithm):
    xtrain, xtest, ytrain, ytest = create_pca(db_name, feature_num, classifier, fs_algorithm)
    xtrain, ytrain = make_augmentation(xtrain, ytrain)

    cf = CLASSIFIERS[classifier].fit(xtrain, ytrain)
    ypred = cf.predict(xtest)
    acc = accuracy_score(ytest, ypred)

    # TODO: Add results to the final csv:


def create_pca(db, feature_num, classifier, fs_algorithm):
    df = pd.read_csv("after_preprocess/" + db + ".csv", header=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, stratify=y)

    selected = FS_ALGORITHMS[fs_algorithm].fit(xtrain, ytrain.values)
    #selected = TODO: The function that returns the top k features out of the selected features.

    pos = np.asarray(range(0, np.size(xtrain, 1)))
    locations = pos[selected == 1]
    xtrain = xtrain.iloc[:, locations].reset_index(drop=True)
    xtest = xtest.iloc[:, locations].reset_index(drop=True)

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

    return xtrain, xtest, ytrain, ytest

def make_augmentation(xtrain, ytrain):
    sm = over_sampling.SMOTE()
    return sm.fit_resample(xtrain, ytrain)

get_best_config()
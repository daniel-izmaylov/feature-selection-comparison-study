from curses import KEY_OPTIONS
from multiprocessing.dummy import Pool
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin 
# import  sklearn.metrics._scorer:
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#import logistic_regression 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFdr
from tempfile import mkdtemp
from joblib import Memory
from shutil import rmtree
import itertools
from sklearn.feature_selection import SelectFdr, chi2, f_classif
from sklearn.metrics import precision_recall_curve
import numpy as np 
import scipy.io
import ReliefF
from skfeature.function.information_theoretical_based import MRMR
from tqdm import tqdm
from multiprocessing.pool import Pool as PoolParent
from multiprocessing import Process, Pool
import time
from sklearn.svm import SVC
from functools import partial
from sklearn.model_selection import LeavePOut
import pandas as pd 
import scipy.io
from sklearn.feature_selection import RFE
from feature_algo.Genetic_FA import Genetic_FA 
from  feature_algo import dssa 
from multiprocessing.pool import ThreadPool
import pickle    
from timeit import default_timer as timer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def get_clf_dict():
    return {'LogisticRegression': LogisticRegression(),
            # 'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True)}
 
class NoDaemonProcess(Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(PoolParent):
    Process = NoDaemonProcess
    
# FS_ALGO_LIST= ["MRMR","SVM"]
FS_ALGO_LIST= ["f_classif","ReliefF"]
# FS_ALGO_LIST= ["ReliefF"]
# FS_ALGO_LIST= ["f_classif","MRMR","ReliefF"]
# FS_ALGO_LIST= ["dssa","f_classif","MRMR","ReliefF"]

# K_OPTIONS= [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]
K_OPTIONS= [50,100]



def run_grid_search(db):
    X= db[:,:-1]
    y= db[:,-1]

    with MyPool(processes=len(FS_ALGO_LIST)) as pool:
        results= pool.map(partial(calculate_per_FS_algo,X=X,y=y)
                                  ,FS_ALGO_LIST)

    results= {k: v for d in results for k, v in d.items()}
                 
    return results           



def calculate_per_FS_algo(fs_algo,X=[],y=[]):
    print("Calculating for {}".format(fs_algo))
    res={}
    clf_list= get_clf_dict().items()
    empty_feature=False
    best_feature,fs_time = feature_selection(fs_algo,X, y)
    res['fs_time']= fs_time
    
    if fs_algo=="ReliefF":
        if best_feature.pvalues_==None:
            empty_feature=True
            best_feature=[]
            best_feature_index=[] 
    if not empty_feature:
        best_feature=best_feature.scores_
        best_feature_index=np.argsort(best_feature)[::-1]
    
    with ThreadPool(processes=len(K_OPTIONS)) as pool:
        temp= pool.map(partial(k_level,X=X,y=y,best_feature=best_feature, best_feature_index=best_feature_index)
                                  ,K_OPTIONS)
    res= {k: v for d in temp for k, v in d.items()}
    res['fs_time']= fs_time

    return res                


    
    #k level
def k_level(k,X,y,best_feature=[], best_feature_index=[]):
    print("k level:",k)
    res={}
    empty_feature= best_feature== []
    clf_list= get_clf_dict().items()

    if empty_feature:
        res['chosen_features']= []
        res['feature_rank']= []
        for clf_name, clf in clf_list:
            res[clf_name]={}
            res[clf_name][0]={}
            res[clf_name][0]["infrence_time"]=0
            res[clf_name][0]["res"]= {'accuracy':0,"MCC":0,"AUC":0,"PR-AUC":0}
        return {k:res}


    k_best= best_feature_index[:k]
    X_k_best= X[:,k_best]
    res['chosen_features']= k_best
    res['feature_rank']=best_feature[k_best]
        
    fold_func= get_fold(X_k_best)
    #fold level 

        # res[i]={}
    for clf_name, clf in clf_list:
        res[clf_name]={}
        if len(X)<=100: #leave one out
            y_test = []
            y_prob=[]
            y_pred=[]
            start= timer()

            for train, test in fold_func.split(X_k_best, y):
                y_test.append(y[test])
                predict_proba= clf.fit(X_k_best[train], y[train]).predict_proba(X_k_best[test])
                y_prob.append(predict_proba[:,1])
                y_pred.append(np.argmax(predict_proba, axis=1))
            infrence_time= timer()-start
            y_test = np.array(y_test)
            y_prob = np.array(y_prob)
            res[clf_name][0]=evalute(y_test,y_pred,y_prob,loo=True)
            res[clf_name][0]["infrence_time"]= infrence_time/len(X)


        else: 
            for i, (train_index, test_index) in enumerate(fold_func.split(X_k_best, y)):
                X_train, X_test = X_k_best[train_index], X_k_best[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf.fit(X_train, y_train)
                y_pred= clf.predict(X_test)
                start= timer()

                y_prob= clf.predict_proba(X_test)
                infrence_time= timer()-start

                res[clf_name][i]=evalute(y_test,y_pred,y_prob)
                res[clf_name][i]["infrence_time"]= infrence_time/len(X)

    return {k:res}



def evalute(y_true,y_pred, y_prob, loo=False):
    if y_prob.shape[1] ==2 or  y_prob.shape[1] ==1:
        # if loo:

        y_prob=y_prob[:,0]
        pr_auc= average_precision_score(y_true, y_prob)
        # auc= roc_auc_score(y_true, y_prob)
        fpr, tpr, thresholds = roc_curve(y_true,y_prob)
        auc_Score = auc(fpr, tpr)


    else:
        np.argmax(y_prob, 0)
        bin_y= label_binarize(y_pred, classes=range(y_prob.shape[1]))
        pr_auc= average_precision_score(bin_y, y_prob, average="micro")
        auc_Score= roc_auc_score(bin_y, y_prob, average='micro')
        
    return  {'accuracy':accuracy_score(y_true,y_pred),
            "MCC":matthews_corrcoef(y_true,y_pred),
            "AUC":auc_Score,
            "PR-AUC":pr_auc
    }


def feature_selection(fs_algo,X_train, y_train):
    start = timer()
    
    if  fs_algo=="f_classif":
        return SelectFdr(score_func=f_classif,alpha=0.1).fit(X_train,y_train),timer()-start
    
    elif fs_algo=="MRMR":
        return SelectKBest(score_func=MRMR.mrmr,k=100).fit(X_train, y_train),timer()-start

    elif fs_algo=="ReliefF":
        return SelectKBest(score_func=ReliefF.ReliefF,k=100).fit(X_train, y_train),timer()-start
    
    elif fs_algo=="SVM":
        return RFE(SVC(kernel='linear'),n_features_to_select=100,step=1).fit(X_train, y_train),timer()-start
    
    elif fs_algo=="Genetic":
            fs_function= Genetic_FA()
            return  SelectKBest(fs_function.fit,k=100).fit(X_train,y_train),timer()-start


    elif fs_algo=="dssa":
            # fs_function= dssa()
            return  SelectKBest(dssa.fit,k=100).fit(X_train,y_train),timer()-start
        
    
    
def get_fold(x):
    if len(x)<50:
        fold_func = LeavePOut(2)
    elif len(x)<=100:
        fold_func = LeavePOut(1)
    elif len(x)<=1000:
        fold_func = StratifiedKFold(n_splits=10)
    else:
        fold_func = StratifiedKFold(n_splits=5) 
    return fold_func



def clf_res(res,clf):
    # res= k_value_dict[k][clf]
    clf_df= pd.DataFrame.from_dict(res)
    clf_df.index = clf_df.index.set_names(['metric'])
    clf_df.reset_index(inplace=True)
    clf_df["Learning algorithm"]= clf
    
    return clf_df

def turn_resDict_to_df(results):
    all_df=[]
    for fold,fs_algo_lst in results.items():
        for fs_algo in fs_algo_lst:
            for fs_algo_name in fs_algo.keys():
                k_value_dict= fs_algo[fs_algo_name]
                k_df= pd.DataFrame()
                fs_time= k_value_dict["fs_time"]
                for k in list(k_value_dict.keys())[1:]:
                
                    res_df= pd.DataFrame()
                    res_df=map(lambda clf:(clf_res(k_value_dict[k][clf],clf)),list(k_value_dict[k].keys())[2:])
                    res_df= pd.concat(res_df)
                    # display(res_df)
                    chosen_features= k_value_dict[k]["chosen_features"]
                    feature_rank= k_value_dict[k]["feature_rank"]
                    res_df["chosen_features"]=[chosen_features]*res_df.shape[0]
                    res_df["Selected Features scores"]=[feature_rank]*res_df.shape[0]
                    res_df["fs_time"]=[fs_time]*res_df.shape[0]
                    res_df["Number of features selected (K)"]=[k]*res_df.shape[0]
                    res_df["Filtering Algorithm"]=[fs_algo_name]*res_df.shape[0]
                    res_df["Fold"]=[fold]*res_df.shape[0]
                    all_df.append(res_df.copy(deep=True))
    return pd.concat(all_df, ignore_index=True)

from  sklearn.model_selection import StratifiedKFold

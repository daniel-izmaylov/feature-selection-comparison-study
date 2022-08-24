from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFdr, chi2, f_classif
from sklearn.metrics import precision_recall_curve
import numpy as np 
import feature_algo.reliefF as reliefF
from skfeature.function.information_theoretical_based import MRMR
from multiprocessing.pool import Pool as PoolParent
from sklearn.svm import SVC
from functools import partial
from sklearn.model_selection import LeavePOut
import pandas as pd 
import scipy.io
from sklearn.feature_selection import RFE
from feature_algo.Genetic_FA import Genetic_FA 
from  feature_algo import dssa 
from  feature_algo import New_dssa 

from multiprocessing.pool import ThreadPool
import pickle    
from timeit import default_timer as timer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from utlis.utlis import MyPool

def get_clf_dict():
    return {'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True)}
 

    
FS_ALGO_LIST= ["dssa","f_classif","MRMR","ReliefF","New_dssa","Genetic","SVM"]
K_OPTIONS= [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]




def run_grid_search(db):
    """
    run grid search on the all the configurations and return the best classifier
    """
    X= db[:,:-1]
    y= db[:,-1]
    # septate each feature selection algorithm into a separate process 
    with MyPool(processes=len(FS_ALGO_LIST)) as pool:
        results= pool.map(partial(calculate_per_FS_algo,X=X,y=y)
                                  ,FS_ALGO_LIST)
    results= {k: v for d in results for k, v in d.items()}
    return results    



def calculate_per_FS_algo(fs_algo,X=[],y=[]):
    """
    calculate the performance of each feature selection algorithm
    """
    print("Calculating for {}".format(fs_algo))
    res={}
    clf_list= get_clf_dict().items()
    empty_feature=False
    best_feature,fs_time = feature_selection(fs_algo,X, y)
    res['fs_time']= fs_time
    
    ranking=False
    if fs_algo=="SVM":
        ranking=True
        best_feature= best_feature.ranking_

    elif fs_algo=="ReliefF":
        best_feature= best_feature

    elif  fs_algo!="SVM" and not empty_feature:
        best_feature=best_feature.scores_
    
    #each K value run on a separate thread
    with ThreadPool(processes=len(K_OPTIONS)) as pool: 
        temp= pool.map(partial(k_level,X=X,y=y,best_feature=best_feature,ranking=ranking)
                                  ,K_OPTIONS)
    res= {k: v for d in temp for k, v in d.items()}
    res['fs_algo']= fs_time

    return {fs_algo:res}                


    
def k_level(k,X,y,best_feature=[],ranking=False):
    """
    calculate the performance of each feature selection algorithm for a given k value
    """
    print("k level:",k)
    res={}
    empty_feature= best_feature== []
    clf_list= get_clf_dict().items()

    if ranking:
        k_best_index= best_feature[:k]
    else:
        k_best_index=np.argpartition(best_feature, -k)[-k:]


    X_k_best= X[:,k_best_index]
    res['chosen_features']= k_best_index
    res['feature_rank']=best_feature[k_best_index]
        
    fold_func= get_fold(X_k_best)


    for clf_name, clf in clf_list: # for each classifier calculate the performance on each fold
        res[clf_name]={}

        if len(X)<=100: #calculation for leave one out 
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
    """
    evaluate the performance of the classifier results
    """
    if y_prob.shape[1] ==2 or  y_prob.shape[1] ==1: #binary classification
        if y_prob.shape[1] ==2:
            y_prob= y_prob[:,1]
        else:
            y_prob= y_prob[:,0]
        pr_auc= average_precision_score(y_true, y_prob)
        fpr, tpr, thresholds = roc_curve(y_true,y_prob)
        auc_Score = auc(fpr, tpr)

    else: #multiclass classification
        np.argmax(y_prob, 0)
        bin_y= label_binarize(y_true, classes=range(y_prob.shape[1]))
        pr_auc= average_precision_score(bin_y, y_prob, average="micro")
        auc_Score= roc_auc_score(bin_y, y_prob, average='micro')
        
    return  {'accuracy':accuracy_score(y_true,y_pred),
            "MCC":matthews_corrcoef(y_true,y_pred),
            "AUC":auc_Score,
            "PR-AUC":pr_auc
    }


def feature_selection(fs_algo,X_train, y_train):
    """
    preform feature selection algorithm on the training set and return the  feature score
    """
    start = timer()
    
    if  fs_algo=="f_classif":
        return SelectFdr(score_func=f_classif,alpha=0.1).fit(X_train,y_train),timer()-start
    
    elif fs_algo=="MRMR":
        return SelectKBest(score_func=MRMR.mrmr,k=100).fit(X_train, y_train),timer()-start

    elif fs_algo=="ReliefF":
        temp=reliefF.reliefF(X_train, y_train,mode="raw",n_features_to_keep=100)
        time= timer()-start
        return temp,time
        # return SelectKBest(score_func=ReliefF.ReliefF,k=100).fit(X_train, y_train),timer()-start
    
    elif fs_algo=="SVM":
        return RFE(SVC(kernel='linear'),n_features_to_select=100,step=1).fit(X_train, y_train),timer()-start
    
    elif fs_algo=="Genetic":
            fs_function= Genetic_FA()
            return  SelectKBest(fs_function.fit,k=100).fit(X_train,y_train),timer()-start
    elif fs_algo=="dssa":
            # fs_function= dssa()
            return  SelectKBest(dssa.fit,k=100).fit(X_train,y_train),timer()-start

    elif fs_algo=="New_dssa":
        return  SelectKBest(New_dssa.fit,k=100).fit(X_train,y_train),timer()-start    
    
    
def get_fold(x):
    """
    return a fold function for cross validation defendant on the number of samples in the dataset
    """
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
    clf_df= pd.DataFrame.from_dict(res)
    clf_df.index = clf_df.index.set_names(['metric'])
    clf_df.reset_index(inplace=True)
    clf_df["Learning algorithm"]= clf
    
    return clf_df


from  sklearn.model_selection import StratifiedKFold
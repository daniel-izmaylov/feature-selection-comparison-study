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
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#import logistic_regression 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

FS_ALGO_LIST= ["f_classif","SelectFdr"]
K_OPTIONS= [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100]

def get_clf_dict():
    return {'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'GaussianNB': GaussianNB(),
            'SVC': SVC(probability=True)}



def my_score(X, y):
    return mutual_info_regression(X, y, random_state=0)

from sklearn.preprocessing import label_binarize

def evalute(y_true,y_pred, y_prob):
    if y_prob.shape[1] ==1:
        multi_class=False
        y_prob=y_prob[:,0]
        pr_auc= average_precision_score(y_true, y_prob)
        auc= roc_auc_score(y_true, y_prob)

    else:
        np.argmax(y_prob, 0)
        bin_y= label_binarize(y_pred, classes=range(y_prob.shape[1]))
        pr_auc= average_precision_score(bin_y, y_prob, average="micro")
        auc= roc_auc_score(bin_y, y_prob, average='micro')
        
    return  {'accuracy':accuracy_score(y_true,y_pred),
            "MCC":matthews_corrcoef(y_true,y_pred),
            "AUC":auc,
            "PR-AUC":pr_auc
    }


def feature_selection(fs_algo,X_train, y_train):
    #TODO: need to add out features, SVM
    start = timer()

    if fs_algo=="f_classif":
        return SelectKBest(k=100).fit(X_train,y_train),  timer()-start
    elif fs_algo=="SelectFdr":
        return SelectFdr(score_func=ReliefF.ReliefF,alpha=0.1).fit(X_train,y_train),timer()-start
    
    elif fs_algo=="MRMR":
        return SelectKBest(score_func=MRMR.mrmr,k=100).fit(X_train, y_train),timer()-start
    
    # elif fs_algo=="SVM":
    #     clf= SVC().fit(X_train,y_train)
    #     clf.

from timeit import default_timer as timer
 
    
    
def calculate_per_FS_algo(fs_algo,X_train=[],X_test=[], y_train=[],y_test=[]):
    res={}
    clf_list= get_clf_dict().items()
    empty_feature=True
    #fix it to all the algorithms
    # best_feature = SelectKBest(fs_algo, k=100).fit(X_train, y_train)
    if fs_algo=="SelectFdr":
        print("here")
    best_feature,fs_time = feature_selection(fs_algo,X_train, y_train)
    res['fs_time']= fs_time
    
    if type(best_feature.pvalues_)== np.ndarray:
        empty_feature=False
        best_feature=best_feature.scores_
        best_feature_index=np.argsort(best_feature)[::-1]
        
        
    #k level
    for k in tqdm(K_OPTIONS):
        res[k]={}


        if empty_feature:
            res[k]['chosen_features']= []
            res[k]['feature_rank']= []
            
        else:
            k_best= best_feature_index[:k]
            X_train_k= X_train[:,k_best]
            X_test_k= X_test[:,k_best]
            res[k]['chosen_features']= k_best
            res[k]['feature_rank']=best_feature[k_best]
            
            
        #classifier level
        for clf_name,clf in clf_list:
            res[k][clf_name]={}
            if empty_feature:
                res[k][clf_name]['proba_time']=0
                res[k][clf_name]["res"]=[]
            else:    
                clf.fit(X=X_train_k,y= y_train)
                start = timer()
                y_proba= clf.predict_proba(X_test_k)
                res[k][clf_name]['proba_time']= timer()-start

                y_pred= clf.predict(X_test_k)
                res[k][clf_name]["res"]=evalute(y_test,y_pred,y_proba)
    return res
    
    
    
    
from functools import partial


def run_grid_seach(db):
    results={}
    #TODO: freeze seed
    #TODO: add all the types of cross validation
    kf = KFold(n_splits=5)
    # result = next(kf.split(db), None)
    clf_list= get_clf_dict().items()
    fs_algo_list= FS_ALGO_LIST
    #cross validation level
    for fold_n,(train, test) in enumerate(kf.split(db)):
        print("Fold: ", fold_n)
        results[fold_n]={}
        train = db[train]
        test =  db[test]
        X_train= train[:,:-1]
        y_train= train[:,-1]
        X_test= test[:,:-1]
        y_test= test[:,-1]
        # feature selection level
        with Pool(processes=len(fs_algo_list)) as pool:
            results[fold_n]= pool.map(partial(calculate_per_FS_algo,X_train=X_train,X_test=X_test, y_train=y_train,y_test=y_test),
                                      fs_algo_list)[0]
            
        print(results)



import scipy.io

mat =scipy.io.loadmat("Data/scikit-Dataset/lymphoma.mat")
X=mat['X']
y = mat['Y'][:, 0]
# attach y to the end of X
db= np.concatenate((X,y.reshape(-1,1)),axis=1)

run_grid_seach(db,'lymphoma')
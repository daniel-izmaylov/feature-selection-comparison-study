
import numpy as np
from numpy.random import rand
from numpy.random import randint
from numpy.random import choice
import random

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# error rate
def error_rate(xt, xv, yt, yv, x):
    # parameters
    k = 5

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain = xt[:, x == 1]
    ytrain = yt.reshape(num_train)  # Solve bug
    xvalid = xv[:, x == 1]
    yvalid = yv.reshape(num_valid)  # Solve bug
    # Training
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred = mdl.predict(xvalid)
    acc = np.sum(yvalid == ypred) / num_valid
    error = 1 - acc

    return error


# Error rate & Feature size
def Fun(xt, xv, yt, yv, x):
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        error = error_rate(xt, xv, yt, yv, x)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost

def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin

def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x

def LSA(fitF, Xbin, xt, xv, yt, yv, max_it = 10):
    t1 = fitF
    k = 1
    while k <= max_it:
        num_features = choice([2, 5])
        feature_numbers = randint(0, Xbin.size, num_features)
        for feature in feature_numbers:
            if Xbin[feature] == 1:
                Xbin[feature] = 0
            else:
                Xbin[feature] = 1
        new_fit = Fun(xt, xv, yt, yv, Xbin)
        if new_fit < t1:
            t1 = new_fit
        k += 1
    return t1


def fit(x, y):
    # Parameters
    ub    = 1
    lb    = 0
    thres = 0.5
    
    N        = 10
    max_iter = 100
    M = random.uniform(0.9, 1.08)
    U_Value = rand()

    xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.3, stratify=y)

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X     = init_position(lb, ub, N, dim)
    
    # Pre
    fit   = np.zeros([N, 1], dtype='float')
    Xf    = np.zeros([1, dim], dtype='float')
    fitF  = float('inf')
    lastF = fitF
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    improvement_counter = 0

    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        for i in range(N):
            fit[i,0] = Fun(xtrain, xval, ytrain, yval, Xbin[i,:])
            if fit[i,0] < fitF:
                Xf[0,:] = X[i,:]
                fitF    = fit[i,0]
                XbinF = Xbin[i, :]

        if fitF == lastF:
            improvement_counter += 1
        else:
            improvement_counter = 0

        if improvement_counter >= 2:
            fitF = LSA(fitF, XbinF, xtrain, xval, ytrain, yval)
            if fitF != lastF:
                improvement_counter = 0

        # Store result
        lastF = fitF
        curve[0,t] = fitF.copy()
        print("Iteration:", t + 1)
        print("Best (SSA):", curve[0,t])
        t += 1
        
 	    # Compute coefficient, c1 (3.2)
        c1 = 2 * np.exp(-(4 * t / max_iter) ** 2)
        
        for i in range(N):          
            # First leader update
            if i == 0:  
                for d in range(dim):
                    # Coefficient c2 & c3 [0 ~ 1]
                    c2 = rand() 
                    c3 = rand()
              	    # Leader update (3.1)
                    if c3 >= 0.5: 
                        X[i,d] = Xf[0,d] + c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                    else:
                        X[i,d] = Xf[0,d] - c1 * ((ub[0,d] - lb[0,d]) * c2 + lb[0,d])
                
                    # Boundary
                    X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d]) 
                
            # Salp update
            elif i >= 1:

                U_Value = M*(7.86 * U_Value - 23.31 * (U_Value ** 2) + 28.75 * (U_Value ** 3) - 13.302875 * (U_Value ** 4))
                if U_Value < 0.5:
                    for d in range(dim):
                        # Salp update by following front salp (3.4)
                        X[i, d] = (X[i, d] + X[i - 1, d]) / 2
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
                else:
                    #Should it include the leader salp?
                    x1 = randint(0, X.shape[0] - 1)
                    x2 = randint(0, X.shape[0] - 1)
                    for d in range(dim):
                        # Salp update using equation 5
                        X[i, d] = (X[x1, d] + X[x2, d] + X[0, d]) / 3
                        # Boundary
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

        

    # Best feature subset
    Gbin       = binary_conversion(Xf, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    ssa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'bin_features': Gbin}
    
    return Gbin
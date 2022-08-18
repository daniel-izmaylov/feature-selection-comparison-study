import random

import numpy as np
import pandas as pd
import scipy
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from FS.dssa import jfs   # change this to switch algorithm
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('../datasets/TOX-171.mat')
feat=mat['X']
label = mat['Y'][:, 0]


# split data into train, validation and test
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.3, stratify=ytrain)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xval, 'yv':yval}

scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xval = scaler.transform(xval)
xtest = scaler.transform(xtest)

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
M = random.uniform(0.9, 1.08)
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'M': M}

# perform feature selection
fmdl = jfs(xtrain, ytrain, opts)
sf   = fmdl['sf']

# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)

# # plot convergence
# curve   = fmdl['c']
# curve   = curve.reshape(np.size(curve,1))
# x       = np.arange(0, opts['T'], 1.0) + 1.0
#
# fig, ax = plt.subplots()
# ax.plot(x, curve, 'o-')
# ax.set_xlabel('Number of Iterations')
# ax.set_ylabel('Fitness')
# ax.set_title('PSO')
# ax.grid()
# plt.show()


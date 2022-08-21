import sys
import numpy as np
import pandas as pd
import comparison
import pickle
import random
from os import walk

#freeze all seeds for reproducibility
import os 
def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main(database_name):
    freeze_seed(42)
    
    db=pd.read_csv("after_preprocess/"+database_name+".csv",header=0)
    results=comparison.run_grid_search(db.values)
    results["Dataset Name"]=database_name
    results["Number of samples"]=db.shape[0]
    results["Original Number of features"]=db.shape[1]-1
    #save results into pickle file
    with open("results/"+database_name+".pkl", "wb") as f:
        pickle.dump(results, f)

from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    #TODO: ours feature selection 
    #TODO: binary classification
    #TODO: names of colum not index
    # task_n= int(os.environ['SLURM_ARRAY_TASK_ID'])
    # file_name= [f for f in listdir("after_preprocess") if isfile(join("after_preprocess", f))][task_n]
    # database_name= file_name.split(".")[0]
    # print(database_name)

    # main(file_name.split(".")[0])

    database_name="ARFF_CNS"
    main(database_name)
from time import sleep
import numpy as np
import pandas as pd
import comparison
import random

#freeze all seeds for reproducibility
import os

from utlis.utlis import turn_resDict_to_df 

def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main(database_name):
    freeze_seed(42)
    
    db=pd.read_csv("after_preprocess/"+database_name+".csv",header=0)
    columns=list(db.columns)

    results=comparison.run_grid_search(db.values)
    results["Dataset Name"]=database_name
    results["Number of samples"]=db.shape[0]
    results["Original Number of features"]=db.shape[1]-1
    #save results as csv file
    results_csv= turn_resDict_to_df(results,columns)
    results_csv.to_csv("results/"+database_name+"_results.csv",index=False)

    #save results into pickle file
    # with open("results/"+database_name+".pkl", "wb") as f:
    #     pickle.dump(results, f)

from os import listdir
from os.path import isfile, join


if __name__ == "__main__":
    try:
        task_n= int(os.environ['SLURM_ARRAY_TASK_ID'])
    except KeyError:
        task_n=4
    files= [f for f in listdir("after_preprocess") if isfile(join("after_preprocess", f))]
    file_name=files[task_n]
    print(file_name.split(".")[0])

    main(file_name.split(".")[0])

    # database_name="bioconductor_bladderbatch"
    # main(database_name)
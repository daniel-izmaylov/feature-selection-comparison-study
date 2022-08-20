import sys
import numpy as np
import pandas as pd
import comparison
import pickle
import random
#freeze all seeds for reproducibility

def freeze_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main(database_name):
    freeze_seed(42)
    db=pd.read_csv("after_preprocess/"+database_name+".csv",header=None)
    results=comparison.run_grid_search(db.values)
    results["Dataset Name"]=database_name
    results["Number of samples"]=db.shape[0]
    results["Original Number of features"]=db.shape[1]-1
    
    #save results into pickle file
    with open("results/"+database_name+".pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    #TODO: ours feature selection 
    #TODO: binary classification
    #TODO: names of colum not index

    
    # database_name= sys.argv[1]
    
    database_name="scikit_COIL20"
    main(database_name)
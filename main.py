import sys
import numpy as np
import pandas as pd
import comparison
import pickle


def main(database_name):
    db=pd.read_csv("after_preprocess/"+database_name+".csv",header=None)
    results=comparison.run_grid_search(db.values)
    results["database"]=database_name
    #save results into pickle file
    with open("results/"+database_name+".pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    #TODO: ours feature selection 
    #TODO: binary classification
    #TODO: names of colum not index
    #TODO: add more db information
    # TODO: add svd feature selection
    #TODO: all seed
    
    # database_name= sys.argv[1]
    database_name="scikit_COIL20"
    main(database_name)
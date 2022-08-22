import pandas as pd 

def get_fold(x):
    if x<50:
        return "Leave-pair-out"
    elif x<=100:
        return "LOOCV"
    elif x<=1000:
        return "StratifiedKFold n_splits = 10"
    else:
        return "StratifiedKFold n_splits = 5"

def turn_resDict_to_df(results,columns):
    all_df=[]
    for algo_name,fs_algo_lst in list(results.items())[:-3]:
        # database_name= results
        for k, k_res in list(fs_algo_lst.items())[:-1]:
            fs_time= fs_algo_lst["fs_algo"]
            # print(fold, fold_res)
            chosen_features= k_res["chosen_features"]
            feature_rank= k_res["feature_rank"]
            # print(fold)
            # print(fold_res.keys())
            for clf_name,clf_res in list(k_res.items())[2:]:
                for fold_name, fold_res in clf_res.items():
                    # print(fold_res)
                    infrence_time=fold_res["infrence_time"]
                    # del fold_res["infrence_time"]
                    temp_d=pd.DataFrame.from_dict(fold_res, orient='index')
                    temp_d.reset_index(inplace=True)
                    temp_d=temp_d.melt(id_vars='index')
                    temp_d.drop(columns=['variable'],inplace=True)
                    temp_d["infrence_time"]=[infrence_time]*temp_d.shape[0]
                    temp_d["Learning algorithm"]=[clf_name]*temp_d.shape[0]
                    temp_d["Number of features selected (K)"]=[k]*temp_d.shape[0]
                    temp_d["chosen_features"]=[chosen_features]*temp_d.shape[0]
                    temp_d["Selected Features scores"]=[feature_rank]*temp_d.shape[0]
                    temp_d["Filtering Algorithm"]=[algo_name]*temp_d.shape[0]
                    temp_d["Fold"]=[fold_name]*temp_d.shape[0]
                    temp_d["fs_time"]=[fs_time]*temp_d.shape[0]
                    all_df.append(temp_d.copy(deep=True))
    all_df=pd.concat(all_df, ignore_index=True)
    all_df["Dataset Name"]=results["Dataset Name"]
    all_df["Number of samples"]=results["Number of samples"]
    all_df["Original Number of features"]=results["Original Number of features"]
    all_df["chosen_features"]= all_df["chosen_features"].apply(lambda x: [columns[i] for i in x])
    all_df["CV Method"]=get_fold(results["Original Number of features"])
    return all_df



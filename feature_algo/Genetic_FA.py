# %%
import pandas as pd 
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# pd.options.plotting.backend = "plotly"
from numpy.random import randint
from numpy.random import rand
import random
import heapq
from tqdm import tqdm
import scipy.io
from sklearn.feature_selection import f_classif
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
from collections import Counter
import statistics
from scipy.stats import chi2_contingency
from tqdm.contrib.concurrent import process_map  # or thread_map
# from sets import Set
from collections import defaultdict
import faiss
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])





def contingency_table(a,b,total):
    obs = np.array([[a,total-a ], [b,total-b]])
    chi2, p, dof, ex =chi2_contingency(obs)
    # print(p)
    if p>0.05:
        return True
    else:
        return False



class Genetic_FA():
    def __init__(self,X_train_kbest_valid, X_valid_kbest_valid, y_train_valid, y_valid_valid,n_features  ) -> None:
        self.X_train_kbest_valid=X_train_kbest_valid
        self.X_valid_kbest_valid=X_valid_kbest_valid
        self.y_train_valid=y_train_valid
        self.y_valid_valid=y_valid_valid
        self.n_features= n_features
        # self.feature_set= 
        #make set of number untill 500
        self.feature_set=set(range(n_features))
        

            
    
    def accuracy(self,y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)
        #use the KNN to compute the number of currectly classified samples/ total samples
        
    def accuracy_knn(self,selected_features):
        # create pipeline
        # pipeline = Pipeline([('scaler', StandardScaler()),
        #                     ('knn', KNeighborsClassifier(n_neighbors=3))])
        # # fit pipeline
        
        # pipeline = Pipeline([('knn', KNeighborsClassifier(n_neighbors=3))])
        # pipeline = Pipeline([('knn', KNeighborsClassifier(n_neighbors=3))])
        # pipeline= FaissKNeighbors(k=3)
        pipeline= KNeighborsClassifier(n_neighbors=3)
        pipeline.fit(self.X_train_kbest_valid[:, selected_features], self.y_train_valid)
        # predict on test set
        y_pred = pipeline.predict(self.X_valid_kbest_valid[:, selected_features])
        # return accuracy
        return self.accuracy(self.y_valid_valid, y_pred)


    def fitness_function(self,bitstring):
        #use the KNN to compute the number of currectly classified samples/ total samples
        return self.accuracy_knn(bitstring)

    def selection(self,population,scores):
        index = random.choices(range(len(population)), k=3)
        high_index = sorted(index, key=lambda agent: scores[agent], reverse=True)[0]
        return population[high_index]


    def r_calculation(self,f_max,f_mead,p1, p2):
        f=max(self.accuracy_knn(p1)
            ,self.accuracy_knn(p2))
        r_cross=1
        if f>f_mead:
            temp=1-((f-f_mead)/(f_max-f_mead))
            r_cross=r_cross*temp
        return r_cross


    def crossover(self,p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1)-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
    
    def calcualte_mutation(self,f_max,f_mead,mut):
        f_mut=self.accuracy_knn(mut)
        r_mut=1.0
        if f_mut>f_mead:
            temp=1-((f_mut-f_mead)/(f_max-f_mead))
            r_mut=r_mut*temp
        return r_mut

    # mutation operator
    def mutation(self,bitstring, r_mut,n_features):
        for i in range(len(bitstring)):
                bitstring_set=set(bitstring)
                if rand() < r_mut:
                    left_number= list(self.feature_set-bitstring_set)
                    bitstring[i]=random.choice(left_number)
                    
    def repopulate(self,pop,n_features):
        scores = [self.fitness_function(c) for c in pop]
        f_mead=np.mean(scores)
        for i in range(len(pop)):
            if scores[i]< f_mead:
                pop[i]= random.sample(range(n_features), 30)
        return pop

    def genetic_algorithm(self, n_generations=1):
        # initial population of random bitstring
        n_pop=50
        pop=[random.sample(range(self.n_features), 30) for i in range(n_pop)]
        
        old_best= pop[0]
        old_best_score=0
        gen_of_best=0
        gen=0
        # enumerate generations
        while True:
            if gen_of_best>=20 and gen>=100 or old_best_score==1.0:
                return old_best
            
            print("Generation:", gen)
            # evaluate all candidates in the population
            if gen>0 and gen%10==0:
                pop= self.repopulate(pop,self.n_features)


            
            scores = [self.fitness_function(c) for c in pop]
            # find the two best candidate
            best_scores= heapq.nlargest(2, range(len(scores)), key=scores.__getitem__)
            best_score=scores[best_scores[0]]



            if best_score>old_best_score:
                old_best_score=best_score
                old_best=pop[best_scores[0]]
                gen_of_best=0


            # tqdm.write("Generation: {}, Best: {}".format(gen, best_score))
            if best_score==1.0:
                # print(">%d, new best f(%s) = %.3f" % (gen,  pop[best_scores[0]],best_score))
                return pop[best_scores[0]]


    
            selected = [self.selection(pop, scores) for _ in range(n_pop-2)]
            # create the next generation
            children = [pop[best_scores[0]],pop[best_scores[1]]]
            f_max=max(scores)
            f_mead=np.mean(scores)


            for i in range(0, n_pop-2, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                r_cross=self.r_calculation(f_max,f_mead,p1, p2)
                for c in self.crossover(p1, p2, r_cross):
                    # mutation
                    r_mut= self.calcualte_mutation(f_max,f_mead,c)
                    self.mutation(c, r_mut,self.n_features)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
            gen_of_best+=1
            gen+=1
        # return best

    
    def fit(self):
        lst=[]
        # for i in tqdm(range(10)):
        #     lst.append(self.genetic_algorithm(i))

        # with Pool(20) as pool:
            # N = pool.map(, range(10))
            # r = list(tqdm(pool.imap(self.genetic_algorithm, range(1000)), total=1000))
        # r = process_map(self.genetic_algorithm, range(0, 10000), max_workers=20)
        r = process_map(self.genetic_algorithm, range(0, 40), max_workers=40)
        #defult dict with set
        counting_dict=defaultdict(set)
        for i,lst in enumerate(r):
            for number in lst:
                counting_dict[number].add(i)
                
        # r= Counter(x for xs in r for x in set(xs))
        # return r
        
        mean = statistics.mean(map(len,counting_dict.values()))
        counting_dict= {k:len(v) for k,v in counting_dict.items() if len(v) > mean}
        counting_dict=dict(sorted(counting_dict.items(), key=lambda item: item[1],reverse=True))
        r_lst= list(counting_dict.keys())
        set_final=set()
        for i,key in enumerate(r_lst[:-1]):
            keys_2=r_lst[i+1]
            if contingency_table(counting_dict[key],counting_dict[keys_2],500):
                set_final.add(key)
                set_final.add(keys_2)
        self.select_k=list(set_final)
        return list(set_final)

import multiprocessing

    
def main():
    n_features = 500
    ga=Genetic_FA(X_train_kbest,X_valid_kbest_valid,y_train_valid, y_valid_valid,n_features)
    r=ga.fit() 
    mat =scipy.io.loadmat('scikit-Dataset/TOX-171.mat')
    X=mat['X']
    y = mat['Y'][:, 0] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,  random_state=40)
    k_best = SelectKBest(f_classif, k=500).fit(X_train, y_train)
    X_train_kbest = k_best.transform(X_train)
    X_test_kbest = k_best.transform(X_test)
    

    X_train_kbest_valid, X_valid_kbest_valid, y_train_valid, y_valid_valid = train_test_split(X_train_kbest, y_train,  test_size=0.3, stratify=y_train, random_state=40)
    scalar = StandardScaler()
    X_train_kbest=scalar.fit_transform(X_train_kbest_valid)
    X_valid_kbest_valid=scalar.transform(X_valid_kbest_valid)
    # clf= RandomForestClassifier()
    clf= KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train_kbest,y_train_valid)
    y_pred= clf.predict(X_valid_kbest_valid)
    print(classification_report(y_valid_valid, y_pred))



    r=list(r)
    #train random forest with selected features
    clf=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=0)

    clf.fit(X_train_kbest[:, r],y_train)

    y_pred= clf.predict(X_test_kbest[:, r])
    print(classification_report(y_test, y_pred))

if __name__=="__main__":
    r=main()

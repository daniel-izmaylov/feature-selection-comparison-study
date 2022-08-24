# ML Assignment 4
Daniel Izmaylov - 205587660  
Lielle Todder - 311508568

## Running Process:
1. All datasets should be in the `Data` folder.
2. Run `data_preprocessing.py` . The file automatically reads all datasets from `Data`, 
runs the preprocess pipeline and outputs the preprocessed datasets into `after_preprocess` folder.
3. In order to run the `main.py` file, which runs the whole feature selection and classification algorithms,
you need to run this file through the department's cluster. It will assign each file from `after_preprocess` a process
to run the whole feature selection task. Alternatively, you can edit the main function in this file to run this program on a specific file.
This stage's output is in `results` folder. Each dataset has its own csv, following the table format from the instructions.
4. In order to run the `augmentation.py` file, you need to run this file through the department's cluster. Alternatively, you can edit the main function in this file to run this program on a specific file.
It outputs a csv with the results to `augmenrataion_res` folder.
5. `FriedmanTest.ipynb` is a python notebook that reads all files from the `results` folder and prepares the data in a `friedman.csv` file
in the correct format to run the test.

## Results Analysis:
All graphs that were presented in the report are saved in the `utils` folder, in `result_analysis.ipynb` and `result_augmentation.ipynb`
files. 

## Paper Implementation

Lee, Chien-Pang, and Yungho Leu. "A novel hybrid feature selection method for microarray data analysis." Applied Soft Computing 11.1 (2011): 208-213.  
Implemented in `feature_algo/Genetic_FA.py`

Tubishat, Mohammad, et al. "Dynamic salp swarm algorithm for feature selection." Expert Systems with Applications 164 (2021): 113873.‏   
Implemented in `feature_algo/dssa.py`
Our improved version can be seen at `feature_algo/New_dssa.py`

Both implementations receive X and y as inputs - a numpy arrays of the dataset and its labels (preferably only the train set,
to do the process correctly). They output either a binary array of selected features, or a scored array of selected features.
All implementations can be activated using their corresponding `fit` function.

    
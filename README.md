# Biased Random Forest

Run
```
python main.py
```

with optional command-line arguments
```
-data (str, default='./diabetes.csv'): path to the input file (diabetes.csv)
-k (int, default=10): number of k neighbors for BRAF
-s (int, default=100): total number of trees
-p (float, default=0.5): the ratio used to define the size of random forest
-n_folds (int, default=10): number of folds for CV
-plot (str, default='./plots'): a path to store plots
```
Summary of results with K = 10, p = 0.5, s = 100
```
Train set Results:
Accuracy: 66.07  Precision: 49.17  Recall: 4.33  AUPRC: 0.63  AUROC: 0.80
Test set Results:
Accuracy: 65.65  Precision: 55.67  Recall: 5.56  AUPRC: 0.65  AUROC: 0.82
```
Plots are stored in `./plots`. 

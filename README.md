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
Summary of results with k = 10, p = 0.5, s = 100
```
Trainset Results:
Accuracy: 67.05  Precision: 60.00  Recall: 6.61  AUPRC: 0.65  AUROC: 0.82
Testset Results:
Accuracy: 65.71  Precision: 66.98  Recall: 5.19  AUPRC: 0.66  AUROC: 0.83
```
Plots are stored in `./plots`. 

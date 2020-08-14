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

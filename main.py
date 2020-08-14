
import os
from os.path import join
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from braf.utils import *
from braf.RandomForest import RandomForestClassifier
from braf.BRAF import BRAF

def main(args):

    # read data as a dataframe
    df_data = pd.read_csv(args.data)

    # create a path to store plots
    path = args.plot
    if not os.path.exists(path):
        os.makedirs(path)

    # define the target column name and minor class
    target_name = 'Outcome'
    minor_class = 1
    print('Percentage of minor class: {:.2f}%'.format(sum(df_data[target_name]==minor_class)/len(df_data)*100))

    # check the distributions and correlations
    sns_plot = sns.pairplot(df_data)
    sns_plot.savefig(join(path, 'sns_plot_org.png'))

    # 80/20 stratified training/test split
    df_train, df_test = split_data(df_data, target_name, minor_class, train_size = 0.8, seed = 1)

    # The columns ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] have zeros as missing values
    # Check how much data is missing for each column
    columns_missing_values = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_missing_values:
        print('Prcentage of missing values at {} : {:.3f}%'.format(col, sum(df_train[col]==0)/len(df_train)*100. ))


    # Let's fill in the missing values using KNN
    # This will fill in the missing values using the KNN from the training dataset
    k = 5
    fill_missing_values_knn(df_train, df_test, k, target_name, columns_missing_values)

    # check the distributions and correlations again
    sns_plot = sns.pairplot(df_train)
    sns_plot.savefig(join(path, 'sns_plot_after_preprocess_train.png'))

    sns_plot = sns.pairplot(df_test)
    sns_plot.savefig(join(path, 'sns_plot_after_preprocess_test.png'))

    # normalized dataset inplace
    # means, stds = normalize_dataset(df_train, df_test, target_name, columns_missing_values)
    means, std = min_max_normalize_dataset(df_train, df_test, target_name, columns_missing_values)


    # now apply BRAF
    # Convert the dataframe into numpy array
    train_set, test_set = df_train.values, df_test.values
    print('\n\nPrepare training')
    print('Percentage of minor class at training: {:.2f}%'.format(sum(train_set[:,-1]==1)/len(train_set)*100))
    print('Percentage of minor class at testing: {:.2f}%'.format(sum(test_set[:,-1]==1)/len(test_set)*100))

    k = args.k
    p = args.p
    s = args.s
    # For a single run
    ##braf = BRAF(data, s, p, k, minor_class, max_depth, min_size, sample_size, n_features)
    #braf = BRAF(train_set, s, p, k, minor_class)
    #braf.fit()

    # Evaluate the model using the cross validation
    n_folds = args.n_folds
    print('\nStart Cross Validation with {} folds'.format(n_folds))
    cv_scores, cv_models = cv_evaluate_algorithm(train_set, BRAF, n_folds, path, s, p, k, minor_class)


    # apply CV models into the test set
    scores_list = []
    prc_list, roc_list = [], []
    for model in cv_models:
        actual = test_set[:, -1]
        scores, prc_values, roc_values = evaluate_algorithm(test_set[:,:-1], actual, model)

        scores_list.append(scores)
        prc_list.append(prc_values)
        roc_list.append(roc_values)

    plot_curves(path, prc_list, {'title': 'PR curve (Test set)', 
                                 'xtitle': 'Recall', 'ytitle': 'Precision', 
                                 'filename':'PRC_test_CV.png'})

    plot_curves(path, roc_list, {'title': 'ROC curve (Test set)', 
                                 'xtitle': 'False Positive Rate', 'ytitle': 'True Positive Rate', 
                                 'filename':'ROC_test_CV.png'})

    for idx, scores in enumerate(cv_scores):
        tr_acc, tr_pre, tr_rec, tr_auprc, tr_auroc = scores # scores of the training set
        te_acc, te_pre, te_rec, te_auprc, te_auroc = scores_list[idx] # scores of the test set
        print(f'CV {idx}:')
        print(f'   Training set: Accuracy {tr_acc*100.:.2f}%, Precision {tr_pre*100.:.2f}%, Recall {tr_rec*100.:.2f}%, AUPRC {tr_auprc: .2f} AUROC {tr_auroc :.2f}')
        print(f'   Testing set: Accuracy {te_acc*100.:.2f}%, Precision {te_pre*100.:.2f}%, Recall {te_rec*100.:.2f}%, AUPRC {te_auprc: .2f} AUROC {te_auroc :.2f}')

    print('\nFinal Results')
    metrics = ['Accuracy', 'Precision', 'Recall', 'AUPRC', 'AUROC']
    cv_scores_train = np.array(cv_scores)
    cv_scores_test = np.array(scores_list)
    cv_scores_train[:, :-2] *= 100.
    cv_scores_test[:, :-2] *= 100.
    print('Trainset Results:')
    print(' '.join( [ f'{k}: {v:.2f} '  for k, v in zip(metrics, cv_scores_train.mean(axis=0))] ))
    print('Testset Results:')
    print(' '.join( [ f'{k}: {v:.2f} '  for k, v in zip(metrics, cv_scores_test.mean(axis=0))] ))
    print('Plots are stored at {}'.format(path))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data', type = str, default='diabetes.csv', help = '<path>/diabetes.csv')
    parser.add_argument('-k', type = int, default = 10, help = 'number of k neighbors')
    parser.add_argument('-s', type = int, default = 100, help = 'total number of trees')
    parser.add_argument('-p', type = float, default = 0.5, help = 'the ratio used to define the size of random forest')
    parser.add_argument('-n_folds', type = int, default = 10, help = 'number of CV folds')

    parser.add_argument('-plot', type = str, default = './plots/', help = 'a path to store plots')

    args = parser.parse_args()

    main(args)



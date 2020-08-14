import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from .RandomForest import RandomForestClassifier
from .KNN import KNN

def split_data(df, target_name, target_class, train_size=0.8, seed = 1):
    '''
    Stratified split the dataset into the train and test sets
    Arguments:
        target_name: str, the column name of target
        target_class: int, the integer value of minor class
        train_size: float, ratio of train / test split
    Returns:
        Dataframe: shuffled train and test dataframes
    '''
    
    df_major = df[df[target_name] != target_class] # major class
    df_minor = df[df[target_name] == target_class] # minor class
    
    # random sample for the train set
    df_major_train = df_major.sample(frac=train_size, random_state = seed)
    df_minor_train = df_minor.sample(frac=train_size, random_state = seed)
    # random sample for the test set
    df_major_test = df_major[~df_major.index.isin(df_major_train.index)]
    df_minor_test = df_minor[~df_minor.index.isin(df_minor_train.index)]
    
    df_train = pd.concat([df_major_train, df_minor_train], ignore_index=True)
    df_test = pd.concat([df_major_test, df_minor_test], ignore_index=True)
    print('Split the dataset into {} for training and {} for testing'.format(len(df_train), len(df_test)))
    
    # shuffle rows 
    return df_train.sample(frac=1), df_test.sample(frac=1)

def normalize_dataset(train, test, target_name, cols_missing_values):
    '''
    Normalized data using mean and std
    '''
    df = train.drop(target_name, axis=1)
    mean, std = {}, {}

    for col in df.columns:
        if col in cols_missing_values:
            mask_train = train[col] == 0
            mask_test = test[col] == 0
        else:
            mask_train = train[col] == np.nan
            mask_test = test[col] == np.nan

        mean[col] = df[~mask_train][col].mean()
        std[col] = df[~mask_train][col].std()
        train.loc[~mask_train, col] = (train[~mask_train][col]-mean[col]) / std[col]
        test.loc[~mask_test, col] = (test[~mask_test][col]-mean[col]) / std[col]

    return mean, std

def min_max_normalize_dataset(train, test, target_name, cols_missing_values):
    '''
    Normalize the data using min and max value of the training set
    '''
    df = train.drop(target_name, axis=1)
    mean, std = {}, {}

    for col in df.columns:
        if col in cols_missing_values:
            mask_train = train[col] == 0
            mask_test = test[col] == 0
        else:
            mask_train = train[col] == np.nan
            mask_test = test[col] == np.nan

        df_train = train[~mask_train][col]
        df_test = test[~mask_test][col]

        mean[col] = df_train.mean()
        std[col] = df_train.std()

        train.loc[~mask_train, col] = (df_train - df_train.min())/(df_train.max()-df_train.min())
        test.loc[~mask_test, col] = (df_test - df_train.min())/(df_train.max()-df_train.min())

    return mean, std

def fill_missing_values_knn(train, test, k, target_name, cols_missing_values, missing_value=0):
    '''
    Fill missing values using KNN
    '''
    
    for col in cols_missing_values:
        # drop Outcome column
        mask = train[col] == missing_value
        mask_idx = mask[mask].index.tolist()

        # drop column that has missing values
        missing = train[mask].drop([target_name,col], axis=1)
        non_missing = train[~mask].drop([target_name,col], axis=1)
        # find k neighbors for training set
        k_neighbors = KNN.get_k_neighbors(missing.values, non_missing.values, k = k)
        k_neighbors = list(zip(mask_idx, k_neighbors))
        
        for idx, k_neighbor in k_neighbors:
            mean = train[train[col]!=0][col].iloc[k_neighbor].mean()
            # train.loc[idx, col] = mean
            if col == 'BMI':
                train.loc[idx, col] = mean
            else:
                train.loc[idx, col] = round(mean)

        # using the training set, find KNN for test set
        mask = test[col] == missing_value
        mask_idx = mask[mask].index.tolist()
        missing = test[mask].drop([target_name, col], axis=1)

        k_neighbors = KNN.get_k_neighbors(missing.values, non_missing.values, k = k)
        k_neighbors = list(zip(mask_idx, k_neighbors))
        
        for idx, k_neighbor in k_neighbors:
            mean = train[train[col]!=0][col].iloc[k_neighbor].mean()
            # test.loc[idx, col] = mean
            if col == 'BMI':
                test.loc[idx, col] = mean
            else:
                test.loc[idx, col] = round(mean)



# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    '''
    Cross validation
    Arguments:
        dataset: ndarray
        n_folds: int, number of folds
    Returns:
        list of ndarray
    '''
    dataset_split = []
    dataset_copy = dataset.copy()
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        indicies = np.random.choice(len(dataset_copy), fold_size, replace=False)
        dataset_split.append(dataset_copy[indicies])
        dataset_copy = np.delete(dataset_copy, indicies, axis=0)
    return dataset_split

def accuracy_metric(actual, predicted):
    '''
    Calculate accuracy percentage
    '''
    return sum(actual==predicted)/float(len(actual))

def precision_and_recall(actual, predicted):
    '''
    Calculate precision and recall
    '''
    counts = Counter(zip(actual, predicted))

    true_pos = counts[(1,1)]
    true_neg = counts[(0,0)]
    false_pos = counts[(0,1)]
    false_neg = counts[(1,0)]

    precision = float(true_pos) / (true_pos + false_pos + 1.e-6)
    recall = float(true_pos) / (true_pos + false_neg + 1.e-6)

    return precision, recall

def tpr_and_fpr(actual, predicted):
    '''
    Calculate sensitivity and specificity
    '''
    counts = Counter(zip(actual, predicted))

    true_pos = counts[(1,1)]
    true_neg = counts[(0,0)]
    false_pos = counts[(0,1)]
    false_neg = counts[(1,0)]

    tpr = float(true_pos) / (true_pos + false_neg + 1.e-6)
    fpr = float(false_pos) / (false_pos + true_neg + 1.e-6)

    return tpr, fpr


def get_auprc(actual, prob):
    '''
    Evaluate precision and recall as a function of threshold
    '''
    thresholds = np.argsort(prob)

    precision_list, recall_list = [], []
    for th in thresholds:
        predicted = (prob>=prob[th]).astype(int)
        precision, recall = precision_and_recall(actual, predicted)
        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list

def get_auroc(actual, prob):
    '''
    Evaluate true positive rate and false positive rate as a function of threshold
    '''
    thresholds = np.argsort(prob)

    tpr_list, fpr_list = [], []
    for th in thresholds:
        predicted = (prob>=prob[th]).astype(int)
        tpr, fpr = tpr_and_fpr(actual, predicted)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return tpr_list, fpr_list

def evaluate_algorithm(test_set, actual, model):
    '''
    Evaluate all metrics
    '''
    predicted = model.predictions(test_set)
    accuracy = accuracy_metric(actual, predicted)
    precision, recall = precision_and_recall(actual, predicted)
    pre_list, recall_list = get_auprc(actual, model.predict_prob(test_set))
    tpr_list, fpr_list = get_auroc(actual, model.predict_prob(test_set))

    auprc = -1 * np.trapz(pre_list, recall_list)
    auroc = -1 * np.trapz(tpr_list, fpr_list)

    return (accuracy, precision, recall, auprc, auroc), (pre_list, recall_list), (tpr_list, fpr_list)

def cv_evaluate_algorithm(dataset, algorithm, n_folds, path, *args):
    '''
    Evaluate an algorithm using a cross validation split
    Returns:
        a list contains accuracy, precision, recall, auprc, auroc
        a list contains models from CV
    '''
    folds = cross_validation_split(dataset, n_folds)
    scores_list = []
    cv_models = []
    prc_list, roc_list = [], []

    for ifold, fold in enumerate(folds):
        # copy CV folds
        train_set = folds[:]
        # remove the testing set from the training
        train_set.pop(ifold)

        train_set = np.concatenate(train_set, axis=0)
        test_set = fold.copy()
        test_set[:,-1] = np.nan # remove the target value

        # build algorithm: random forest
        RF = algorithm(train_set, *args)
        RF.fit()

        # test with test_set in CV
        actual = fold[:, -1]
        scores, prc_values, roc_values = evaluate_algorithm(test_set, actual, RF)

        scores_list.append(scores)
        cv_models.append(RF)

        prc_list.append(prc_values)
        roc_list.append(roc_values)

    plot_curves(path, prc_list, {'title': 'PR curve (Train set)', 
                                 'xtitle': 'Recall', 'ytitle': 'Precision', 
                                 'filename':'PRC_train_CV.png'})

    plot_curves(path, roc_list, {'title': 'ROC curve (Train set)', 
                                 'xtitle': 'False Positive Rate', 'ytitle': 'True Positive Rate', 
                                 'filename':'ROC_train_CV.png'})

    return scores_list, cv_models


def plot_curves(path, data_list, plt_config):
    fig, ax = plt.subplots()

    for y, x in data_list:
        auc = -1 * np.trapz(y, x)
        ax.plot(x, y, '--', lw=2, label='AUC {:.2f}'.format(auc), marker='o', clip_on=False)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(plt_config['xtitle'])
    ax.set_ylabel(plt_config['ytitle'])
    ax.set_title(plt_config['title'])
    ax.legend()
    #ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    fig.savefig(os.path.join(path, plt_config['filename']))
    
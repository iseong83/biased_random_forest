# Random Forest
# ref: https://machinelearningmastery.com/implement-random-forest-scratch-python/

import numpy as np
from random import seed
from random import randrange
from collections import Counter

class ForestBase:
    '''
    Base of Forest
    Define the methods commonly used in RandomForest and BRAF
    Methods: predict, predictions, bagging_predict, predict_prob
    '''
    def __init__(self, *args):
        pass

    def predict(self, node, row):
        '''
        Make a prediction with a decision tree
        '''
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def predictions(self, dataset):
        '''
        Make predictions for entire dataset
        '''
        return np.array([self.bagging_predict(self.trees, row) for row in dataset])
        
    def bagging_predict(self, trees, row):
        '''
        Make a prediction with a list of bagged trees
        '''
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def predict_prob(self, dataset):
        '''
        Evaluate probabilty and returns a ndarray
        '''
        probs = []
        for row in dataset:
            predictions = [self.predict(tree, row) for tree in self.trees]
            probs.append(float(predictions.count(1))/len(predictions))
        return np.array(probs)

class RandomForestClassifier(ForestBase):
    '''
    Random Forest Module
    X : ndarray, dataset including binary labels
    max_depth: int, depth of tree to build
    sample_size: float, ratio of random subsample
    n_trees: int, number of trees to build
    n_features: int, maximum number of features 
    '''
    def __init__(self, X, n_trees, max_depth, min_size, sample_size, n_features):
        self.trees = []
        self.X = X
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size =  sample_size
        self.n_features = n_features

    def fit(self):
        '''
        Build Trees
        '''
        print('Creating a forest...')
        print('sample size {}, maximum depth {}, number of tress {}, number of features {}'.format(
            self.sample_size, self.max_depth, self.n_trees, self.n_features))

        for i in range(self.n_trees):
            sample = self.subsample(self.X, self.sample_size)
            tree = self.build_tree(sample, self.max_depth, self.min_size, self.n_features)
            self.trees.append(tree)
    
    def subsample(self, dataset, ratio):
        '''
        Create a random subsample from the dataset with replacement
        Returns a subsample
        '''
        n_sample = round(len(dataset) * ratio)
        indicies = np.random.randint(low=0, high=len(dataset), size=n_sample)
        return dataset[indicies].tolist()
 
    def to_terminal(self, group):
        '''
        Create a terminal node value
        '''
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
 
    def split(self, node, max_depth, min_size, n_features, depth):
        '''
        Create child splits for a node or make terminal
        '''
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, n_features)
            self.split(node['left'], max_depth, min_size, n_features, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, n_features)
            self.split(node['right'], max_depth, min_size, n_features, depth+1)
    
    def build_tree(self, train, max_depth, min_size, n_features):
        '''
        Build a decision tree
        '''
        root = self.get_split(train, n_features)
        self.split(root, max_depth, min_size, n_features, 1)
        return root
    
    def get_split(self, dataset, n_features):
        '''
        Select the best split point for a dataset
        '''
        class_values = [set(row[-1] for row in dataset)]
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = []
        while len(features) < n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
        for index in features:
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def gini_index(self, groups, classes):
        '''
        Calculate the Gini index for a split dataset
        '''
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def test_split(self, index, value, dataset):
        '''
        Split a dataset based on an attribute and an attribute value
        '''
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
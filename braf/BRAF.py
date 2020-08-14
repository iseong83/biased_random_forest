from .KNN import KNN
from .RandomForest import ForestBase, RandomForestClassifier
import numpy as np


class BRAF(ForestBase):
    '''
    Biased Random Forest
    Arguments:
        dataset: ndarray, input data for the training, the last row should be the target
        s: int, size of the generated random forest
        p: float, ratio used to define the size of RF1 and RF2
        minor_class: int, value for minor class
    '''
    def __init__(self, dataset, s, p, k, minor_class, *args):
        self.data = dataset
        self.s = s
        self.p = p
        self.k = k
        self.minor_class = minor_class
        self.args = args

    def fit(self):
        '''
        Fit to the data
        '''

        assert len(self.data.shape) == 2

        dataset = self.data.copy()
        k = self.k

        if not self.args:
            # set default args
            # max depth, min size, sample size, n_features
            self.args = (5, 1, 1.0, len(dataset[0])-1)

        minor_data = dataset[dataset[:,-1]==self.minor_class]
        major_data = dataset[dataset[:,-1]!=self.minor_class]
        print('Number of major class: {}, number of minor class: {}'.format(len(major_data), len(minor_data)))

        k_neighbors = KNN.get_k_neighbors(minor_data, major_data, k = k)
        critical_data = major_data[list(set(k_neighbors.flatten()))]
        critical_data = np.concatenate([minor_data, critical_data], axis=0)
        print('Defined and stored critical dataset: {}'.format(len(critical_data)))
        
        # build two random forests
        print('Creating a first forest on the full data set of size {}'.format(int(self.s*(1.0-self.p))))
        RF1 = RandomForestClassifier(dataset, int(self.s*(1.0-self.p)), *self.args)
        RF1.fit()
        print('Creating a second forest on the critical data set of size {}'.format(int(self.s*self.p)))
        RF2 = RandomForestClassifier(dataset, int(self.s*self.p), *self.args)
        RF2.fit()

        # Combine the two forests to generate the main forest
        self.trees = RF1.trees + RF2.trees




import statistics

import numpy as np
from model_cv import Model_cv
from dataloader import DataLoader
from sklearn.model_selection import RepeatedKFold, KFold

class Train_cv:
    """Training machine learning-based models
    Input:
    1. cv_inner: k-fold cross validation
    2. cv_outer: repeated k-fold cross validation
    3. predictors: one-hot-encoded predictors
    4. outcome: binary
    5. kf: k-fold cross validation split, only needed for naive bayes with missing values
    6. model_type: 'full' or 'simple', only needed for naive bayes with missing values


    Output:
    Printing validation results for each model using each data set.
    """
    def __init__(self, cv_inner, cv_outer, X, y, kf=None, model_type=None):
        self.X = X
        self.y = y
        self.cv_inner = cv_inner
        self.cv_outer = cv_outer
        self.model_type = model_type
        self.kf = kf

    def metric_list(self):
        return ['recall', 'precision', 'accuracy', 'roc_auc']

    def train_nested(self):
        """"Print results for each metric.
        """
        metric_list = self.metric_list()
        for metric in metric_list:
            Model_cv().model_nested_cv(self.cv_inner, self.cv_outer, self.X, self.y, metric=metric)

    def print_bn_results(self):
        print('Naive Bayes (missing values)')
        recall_list, precision_list, acc_list, rocauc_list = Model_cv().nb_cv(self.X, self.y, self.kf, model_type=self.model_type)
        print(': %.3f (%.3f)' % (np.mean(recall_list), statistics.stdev(recall_list)))
        print(': %.3f (%.3f)' % (np.mean(precision_list), statistics.stdev(precision_list)))
        print(': %.3f (%.3f)' % (np.mean(acc_list), statistics.stdev(acc_list)))
        print(': %.3f (%.3f)' % (np.mean(rocauc_list), statistics.stdev(rocauc_list)))

if __name__ == '__main__':
    full_complete_path = 'path_to_file'
    full_missing_path = 'path_to_file'
    full_imputed_path = 'path_to_file'
    simple_complete_path = 'path_to_file'
    simple_missing_path = 'path_to_file'
    simple_imputed_path = 'path_to_file'

    file_type_list = ['full_complete', 'full_imputed', 'simple_complete',
                      'simple_imputed']
    file_path_list = [full_complete_path, full_imputed_path, simple_complete_path, simple_imputed_path]
    model_type_list = ['full', 'full', 'simple', 'simple']

    # machine learning models: complete and imputed data sets
    seed = 1
    num_splits = 10
    num_repeats = 5
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
    cv_outer = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

    # full and simple models
    for i in range(len(file_path_list)):
        file_type = file_type_list[i]
        print(file_type)
        X,y = DataLoader(file_path_list[i], file_type=file_type).get_data(model_type=model_type_list[i])
        Train_cv(cv_inner, cv_outer, X, y).train_nested()

    # naive bayes: data with missing values
    kf = RepeatedKFold(n_splits=10, random_state=5, n_repeats=5)

    # full models
    print('full_missing')
    X,y = DataLoader(full_missing_path, file_type='full_missing').get_data_nb()
    Train_cv(cv_inner, cv_outer, X, y, kf=kf, model_type='full').print_bn_results()

    # simple models
    print('simple_missing')
    X,y = DataLoader(simple_missing_path, file_type='simple_missing').get_data_nb()
    Train_cv(cv_inner, cv_outer, X, y, kf=kf, model_type='simple').print_bn_results()

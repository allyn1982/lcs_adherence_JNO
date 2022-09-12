import numpy as np
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, \
     precision_recall_curve, auc
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
import statistics
import xgboost
from dataloader import DataLoader
from utils import get_var_dict


def model_nested_cv(X, y, cv_inner, cv_outer, metric=None):
    """Function to cross validate machinln learning-based models for complete case and imputed data sets.

    Input:
    X: one-hot-encoded predictors
    y: binary
    cv_inner: k-fold cross validation
    cv_outer: repeated k-fold cross validation
    metric: evaluation metric, i.e., 'recall', 'precision', 'accuracy', 'roc_auc'

    Output:
    Printing validation results for each model using each data set.
    """
    # Complete case and imputed data sets
    # Logistic Regression
    param_grid_lr = {"C": np.logspace(-3, 3, 7)}
    model_lr = LogisticRegression(max_iter=10000000000)
    search_lr = GridSearchCV(model_lr, param_grid_lr, cv=cv_inner)
    scores_lr = cross_val_score(search_lr, X, y, scoring=metric, cv=cv_outer, n_jobs=-1)
    print('Logistic Regression: ', metric, ': %.3f (%.3f)' % (mean(scores_lr), std(scores_lr)))

    # Random Forest
    model_rf = RandomForestClassifier()
    # define search space
    param_grid_rf = {'n_estimators': [10, 100], 'max_features': ['auto', 'sqrt']}
    # define search
    search_rf = GridSearchCV(model_rf, param_grid_rf, n_jobs=1, cv=cv_inner)
    # execute the nested cross-validation
    scores_rf = cross_val_score(search_rf, X, y, scoring=metric, cv=cv_outer, n_jobs=-1)
    print('Random Forest: ', metric, ': %.3f (%.3f)' % (mean(scores_rf), std(scores_rf)))

    # SVM
    param_grid_svm = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
    model_svm = svm.SVC()  # default kernel is rbf
    search_svm = GridSearchCV(model_svm, param_grid_svm, cv=cv_inner)
    scores_svm = cross_val_score(search_svm, X, y, scoring=metric, cv=cv_outer, n_jobs=-1)
    print('SVM: ', metric, ': %.3f (%.3f)' % (mean(scores_svm), std(scores_svm)))

    # Naive Bayes
    model_nb = CategoricalNB()
    scores_nb = cross_val_score(model_nb, X, y, scoring=metric, cv=cv_outer, n_jobs=-1)
    print('Naive Bayes: ', metric, ': %.3f (%.3f)' % (mean(scores_nb), std(scores_nb)))

    # XGBoost
    model_xg = xgboost.XGBClassifier()
    param_grid_xgb = {'model__max_depth': [2, 3, 5], 'model__n_estimators': [10, 50]}
    pipeline = Pipeline([('standard_scaler', StandardScaler()), ('model', model_xg)])
    search = GridSearchCV(pipeline, param_grid_xgb, cv=cv_inner, n_jobs=-1)
    scores_xg = cross_val_score(search, X, y, scoring=metric, cv=cv_outer, n_jobs=-1)
    print('XGBoost: ', metric, ': %.3f (%.3f)' % (mean(scores_xg), std(scores_xg)))


def train_nested(X, y, cv_inner, cv_outer):
    """"Print results for each metric, data set, and model type.
    """
    metric_list = ['recall', 'precision', 'accuracy', 'roc_auc']
    for metric in metric_list:
        model_nested_cv(X, y, cv_inner, cv_outer, metric=metric)

def build_nb_model(model_type=None):
    """Function to create Naive Bayes models.
    Input:
    model_type: 'full' or 'simple'

    Output:
    A Naive Bayes model.

    Note:
    Do not use NaiveBayes() as NaiveBayes().fit() does not have complete_samples_only,
    which needs to be set as False to include missing values
    """
    if model_type == 'full':
        nb_model = BayesianNetwork([('adherence_altered', 'lungrads_12_3_4'),
                                    ('adherence_altered', 'education_new'),
                                    ('adherence_altered', 'median_income_category_new'),
                                    ('adherence_altered', 'fam_hx_lc_new'),
                                    ('adherence_altered', 'comorbid_category_new'),
                                    ('adherence_altered', 'department_new')])
    else:  # simple model
        nb_model = BayesianNetwork([('adherence_altered', 'lungrads_12_3_4'),
                                    ('adherence_altered', 'department_new')])
    return nb_model


def get_feature_cols(model_type):
    """Function to get a list of feature names.
    Input:
    model_type: 'full' or 'simple'

    Output:
    A list of feature names for full or simple model.
    """
    if model_type == 'full':
        feature_cols = ['lungrads_12_3_4',
                        'education_new', 'median_income_category_new', 
                        'fam_hx_lc_new', 'comorbid_category_new', 
                        'department_new'']
    elif model_type == 'simple':
        feature_cols = ['lungrads_12_3_4', 'department_new']
    return feature_cols


def nb_cv(X, y, kf, model_type=None):
    """Function to cross validate Naive Bayes model using data sets with missing values.
    Input:
    predictors: a Pandas data frame of predictors
    outcome: a 1-d numpy array
    kf: k-fold cross validation split
    model_type: 'full' or 'simple'

    Output:
    Print evaluation metrics on validation sets.
    """
    recall_list, precision_list, acc_list, prauc_list, rocauc_list = [], [], [], [], []
    feature_cols = get_feature_cols(model_type=model_type)
    # print(feature_cols)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # add the outcome node (i.e., y_train) to X_train so that the data matrix contains data on all nodes
        X_train['adherence_altered'] = y_train
        # fit model
        nb_model = build_nb_model(model_type=model_type)
        nb_model.fit(X_train, complete_samples_only=False)  # set complete_samples_only=False to include missing values
        # inference
        infer = VariableElimination(nb_model)
        if nb_model.check_model():
            prediction_list = []
            for i in range(len(X_test)):
                # print('Testing:', i+1, '/', len(X_test))
                test_case = list(X_test.iloc[i, :].values)
                # get prediction
                var_dict = get_var_dict(feature_cols, test_case)
                prediction = infer.map_query(['adherence_altered'], evidence=var_dict)['adherence_altered']
                prediction_list.append(prediction)
                # print('predicted: ', prediction, 'truth: ', truth)
            # calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, prediction_list).ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            rocauc = roc_auc_score(y_test, prediction_list)

            recall_list.append(recall)
            precision_list.append(precision)
            acc_list.append(acc)
            rocauc_list.append(rocauc)
            # print('recall: ', recall)
            # print('precision:', precision)
            # print('acc: ', acc)
            # print('rocauc: ', rocauc)
            # print('------------------------')
    return recall_list, precision_list, acc_list, rocauc_list


def print_bn_results(X, y, kf, model_type=None):
    """"Function to print Naive Bayes results for data with missing values.
    """
    print('Naive Bayes (missing values)')
    recall_list, precision_list, acc_list, rocauc_list = nb_cv(X, y, kf, model_type=model_type)
    print('Recall: %.3f (%.3f)' % (np.mean(recall_list), statistics.stdev(recall_list)))
    print('Precision: %.3f (%.3f)' % (np.mean(precision_list), statistics.stdev(precision_list)))
    print('Accuracy: %.3f (%.3f)' % (np.mean(acc_list), statistics.stdev(acc_list)))
    print('ROC-AUC: %.3f (%.3f)' % (np.mean(rocauc_list), statistics.stdev(rocauc_list)))

def build_final_model(classifier=None, model_type=None, file_path=None, file_type=None):
    """Function to build final models.
    Input:
    classifier: specify the classifier
    model_type: 'full' or 'simple'
    file_path: path to training data
    file_type: 'full_imputed' or 'simple_complete'

    Output:
    A final full or simple model.
    """
    final_model = classifier
    # read final training data
    final_train_X, final_train_y = DataLoader(file_path=file_path).get_data(model_type=model_type)
    final_model.fit(final_train_X, final_train_y)
    return final_model

def test_results(classifier=None,  model_type=None, file_path=None, file_type=None, test_path=None):
    """Function to get results on test data.
    Input:
    classifier: specify the classifier
    model_type: 'full' or 'simple'
    file_path: path to training data
    file_type: 'full_complete' or 'simple_imputed'
    test_path: path to hold out test data

    Output:
    Testing results for each metric.
    """
    # read test data
    final_test_X, final_test_y = DataLoader(file_path=file_path).get_test_data(model_type=model_type, test_path=test_path)
    # build final model
    model = build_final_model(classifier=classifier, model_type=model_type, file_path=file_path,
                                   file_type=file_type)
    final_pred = model.predict(final_test_X)
    final_pred_prob = model.predict_proba(final_test_X)[::, 1]
    final_recall = recall_score(final_test_y, final_pred)
    final_precision = precision_score(final_test_y, final_pred)
    final_acc = accuracy_score(final_test_y, final_pred)
    final_auc = roc_auc_score(final_test_y, final_pred_prob)
    print('final_recall', ': %.3f' % (final_recall))
    print('final_precision', ': %.3f' % (final_precision))
    print('final_acc', ': %.3f' % (final_acc))
    print('final_auc', ': %.3f' % (final_auc))

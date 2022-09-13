import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix, \
     precision_recall_curve, auc
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pgmpy.inference import VariableElimination
import statistics

seed = 1
num_splits = 3
num_repeats = 5
cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
cv = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=seed)

def read_data(path):
    return pd.read_csv(path)

def create_var_df(data):
    return data[['lungrads_12_3_4',
                 'education_new',
                 'fam_hx_lc_new',
                 'comorbid_category_new',
                 'median_income_category_new',
                 'department_new']]

def train_data(data, var_df):
    # one-hot-encode non-ordinal and non-binary columns
    # label-encode ordinal and binary columns

    var_list = list(var_df.columns)
    enc = LabelEncoder()
    # transform x
    for var in var_list:
        var_df[var] = enc.fit_transform(var_df[var])
    y = data[['adherence_altered']].values.flatten()
    return var_df, y

def model_nested_cv(predictors, outomce, metric='recall'):
    # logistic regression
    print('logistic regression')
    grid = {"C": np.logspace(-3, 3, 7)}
    model_logit = LogisticRegression()
    model_logit_cv = GridSearchCV(model_logit, grid, cv=cv_inner)
    scores_logit = cross_val_score(model_logit_cv, predictors, outomce, scoring=metric, cv=cv, n_jobs=-1)
    print(metric, ': %.3f (%.3f)' % (mean(scores_logit), std(scores_logit)))

    # random forest
    print('random forest')
    model_rf = RandomForestClassifier()
    # define search space
    space = dict()
    space['n_estimators'] = [10, 100]
    space['max_features'] = ['auto', 'sqrt']
    # define search
    search = GridSearchCV(model_rf, space, n_jobs=1, cv=cv_inner)
    # execute the nested cross-validation
    socres_rf = cross_val_score(search, predictors, outomce, scoring=metric, cv=cv, n_jobs=-1)
    print(metric, ': %.3f (%.3f)' % (mean(socres_rf), std(socres_rf)))

    # SVM
    print('SVM')
    # Set up possible values of parameters to optimize over
    p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
    model_svm = svm.SVC() #default kernel is rbf
    search = GridSearchCV(model_svm, p_grid, cv=cv_inner)
    socres_svm = cross_val_score(search, predictors, outomce, scoring=metric, cv=cv, n_jobs=-1)
    print(metric, ': %.3f (%.3f)' % (mean(socres_svm), std(socres_svm)))

    # naive bayes
    print('naive bayes')
    model_nb = CategoricalNB()
    socres_nb = cross_val_score(model_nb, predictors, outomce, scoring=metric, cv=cv, n_jobs=-1)
    print(metric, ': %.3f (%.3f)' % (mean(socres_nb), std(socres_nb)))

    # XGBoost
    print('xgboost')
    model_xg = xgb.XGBClassifier()
    param_grid = {
        'model__max_depth': [2, 3, 5],
        'model__n_estimators': [10, 50],
    }
    pipeline = Pipeline([
        ('standard_scaler', StandardScaler()),
        ('model', model_xg)
    ])
    search = GridSearchCV(pipeline, param_grid, cv=cv_inner, n_jobs=-1)
    socres_xg = cross_val_score(search, predictors, outomce, scoring=metric, cv=cv, n_jobs=-1)
    print(metric, ': %.3f (%.3f)' % (mean(socres_xg), std(socres_xg)))


def train_nested(X, y):
    model_nested_cv(X, y, metric='recall')
    model_nested_cv(X, y, metric='precision')
    model_nested_cv(X, y, metric='accuracy')
    model_nested_cv(X, y, metric='roc_auc')


def get_var_dict(feature_cols, test_case):
    evidence_dict = {}
    for i in range(len(feature_cols)):
        feature_name = feature_cols[i]
        feature_value = test_case[i]
        if str(feature_value) != 'nan':
            evidence_dict[feature_name] = feature_value
    return evidence_dict

def nb_cv(X, y, nb_model, kf, feature_cols=None):
    recall_list, precision_list, acc_list, prauc_list, rocauc_list = [], [], [], [], []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        # add label to X_train
        X_train['adherence_altered'] = y_train.to_list()
        # fit model
        nb_model.fit(X_train, complete_samples_only=False) # only BayesianNetwork().fit() can specify complete_samples_only, NavieBayes().fit() can't
        # inference
        infer = VariableElimination(nb_model)
        if nb_model.check_model():
            prediction_list = []
            for i in range(len(X_test)):
                #print('Testing:', i+1, '/', len(X_test))
                test_case = list(X_test.iloc[i,:].values)
                # get prediction
                var_dict = get_var_dict(feature_cols, test_case)
                prediction = infer.map_query(['adherence_altered'], evidence=var_dict)['adherence_altered']
                prediction_list.append(prediction)
                #print('predicted: ', prediction, 'truth: ', truth)
            # calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, prediction_list).ravel()
            recall = tp / (tp + fn)
            precision = tp / (tp+fp)
            acc = (tp+tn)/(tp+tn+fp+fn)
            precisions, recalls, thresholds = precision_recall_curve(y_test, prediction_list)

            prauc = auc(recalls, precisions)
            rocauc = roc_auc_score(y_test, prediction_list)

            recall_list.append(recall)
            precision_list.append(precision)
            acc_list.append(acc)
            prauc_list.append(prauc)
            rocauc_list.append(rocauc)
            print('recall: ', recall)
            print('precision:', precision)
            print('acc: ', acc)
            print('prauc: ', prauc)
            print('rocauc: ', rocauc)
            print('------------------------')
    return recall_list, precision_list, acc_list, rocauc_list

def print_bbm_results(recall_list, precision_list, acc_list, rocauc_list):
    print('average recall', ': %.3f (%.3f)' % (np.mean(recall_list), statistics.stdev(recall_list)))
    print('average precision', ': %.3f (%.3f)' % (np.mean(precision_list), statistics.stdev(precision_list)))
    print('average acc', ': %.3f (%.3f)' % (np.mean(acc_list), statistics.stdev(acc_list)))
    print('average rocauc', ': %.3f (%.3f)' % (np.mean(rocauc_list), statistics.stdev(rocauc_list)))

def build_final_model(data, classifier):
    final_model = classifier
    final_var_df = create_var_df(data)
    final_train_X, final_train_y = train_data(data, final_var_df)
    final_model.fit(final_train_X, final_train_y)
    return final_model

def test_X_y(data_path):
    test_data = read_data(data_path)
    test_var_df = create_var_df(test_data)
    final_test_X, final_test_y = train_data(test_data, test_var_df)
    return final_test_X, final_test_y

def test_results(model, final_test_X, final_test_y):
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



import numpy as np
from numpy import std, mean
from pgmpy.inference.ExactInference import VariableElimination
from pgmpy.models.BayesianNetwork import BayesianNetwork
from sklearn.metrics._classification import confusion_matrix
from sklearn.metrics._ranking import roc_auc_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.pipeline import Pipeline
from utils import get_var_dict

class Model_cv:
    """This is a Model class for nested cross validation on using complete case and imputed data sets.
    Input:
    None
    Output:
    Printing scores for each model.
    scores include recall, precision, accuracy, ROC-AUC.
    Models are logistic regression, random forest, SVM, naive Bayes, and XGBoost.
    """
    def model_nested_cv(self, cv_inner, cv_outer, predictors, outcome, metric=None):
        # Complete case and imputed data sets
        # Logistic Regression
        param_grid_lr = {"C": np.logspace(-3, 3, 7)}
        model_lr = LogisticRegression(max_iter=10000000000)
        search_lr = GridSearchCV(model_lr, param_grid_lr, cv=cv_inner)
        scores_lr = cross_val_score(search_lr, predictors, outcome, scoring=metric, cv=cv_outer, n_jobs=-1)
        print('Logistic Regression: ', metric, ': %.3f (%.3f)' % (mean(scores_lr), std(scores_lr)))

        # Random Forest
        model_rf = RandomForestClassifier()
        # define search space
        param_grid_rf = {'n_estimators': [10, 100], 'max_features': ['auto', 'sqrt']}
        # define search
        search_rf = GridSearchCV(model_rf, param_grid_rf, n_jobs=1, cv=cv_inner)
        # execute the nested cross-validation
        scores_rf = cross_val_score(search_rf, predictors, outcome, scoring=metric, cv=cv_outer, n_jobs=-1)
        print('Random Forest: ', metric, ': %.3f (%.3f)' % (mean(scores_rf), std(scores_rf)))

        # SVM
        param_grid_svm = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
        model_svm = svm.SVC()  # default kernel is rbf
        search_svm = GridSearchCV(model_svm, param_grid_svm, cv=cv_inner)
        scores_svm = cross_val_score(search_svm, predictors, outcome, scoring=metric, cv=cv_outer, n_jobs=-1)
        print('SVM: ', metric, ': %.3f (%.3f)' % (mean(scores_svm), std(scores_svm)))

        # Naive Bayes
        model_nb = CategoricalNB()
        scores_nb = cross_val_score(model_nb, predictors, outcome, scoring=metric, cv=cv_outer, n_jobs=-1)
        print('Naive Bayes: ', metric, ': %.3f (%.3f)' % (mean(scores_nb), std(scores_nb)))

        # XGBoost
        model_xg = xgb.XGBClassifier()
        param_grid_xgb = {'model__max_depth': [2, 3, 5], 'model__n_estimators': [10, 50]}
        pipeline = Pipeline([('standard_scaler', StandardScaler()), ('model', model_xg)])
        search = GridSearchCV(pipeline, param_grid_xgb, cv=cv_inner, n_jobs=-1)
        scores_xg = cross_val_score(search, predictors, outcome, scoring=metric, cv=cv_outer, n_jobs=-1)
        print('XGBoost: ', metric, ': %.3f (%.3f)' % (mean(scores_xg), std(scores_xg)))

    def nb_model(self, model_type):
        """Function to create Naive Bayes models.
        Input: 
        model_type: 'full' or 'simple'
        
        Output:
        A Naive Bayes model.

        Note:
        Do not use NaiveBayes() as NaiveBayes().fit() does not have complete_samples_only,
        which needs to be set as False to include observations with missing values
        """
        if model_type == 'full':
            nb_model = BayesianNetwork([('adherence_altered', 'lungrads_12_3_4'),
                      ('adherence_altered', 'age_new'),
                      ('adherence_altered', 'sex_new'),
                      ('adherence_altered', 'race_ethnicity_new'),
                      ('adherence_altered', 'education_new'),
                      ('adherence_altered', 'median_income_category_new'),
                      ('adherence_altered', 'smoking_status_new'),
                      ('adherence_altered', 'fam_hx_lc_new'),
                      ('adherence_altered', 'comorbid_category_new'),
                      ('adherence_altered', 'insurance_new'),
                      ('adherence_altered', 'covid_expected_fu_date_lungrads_interval_new'),
                      ('adherence_altered', 'department_new'),
                      ('adherence_altered', 'distance_to_center_category_new'),
                      ('adherence_altered', 'adi_category_new')])
        else: # simple model
            nb_model = BayesianNetwork([('adherence_altered', 'lungrads_12_3_4'),
                             ('adherence_altered', 'department_new')])
        return nb_model


    def get_feature_cols(self, model_type):
        """Function to get a list'of feature names.
        Input:
        model_type: 'full' or 'simple'
        
        Output:
        A list of feature names for full or simple model.
        """
        if model_type == 'full':
            feature_cols = ['lungrads_12_3_4', 'age_new', 'sex_new', 'race_ethnicity_new',
                    'education_new', 'median_income_category_new', 'smoking_status_new',
                    'fam_hx_lc_new', 'comorbid_category_new', 'insurance_new',
                    'covid_expected_fu_date_lungrads_interval_new', 'department_new',
                    'distance_to_center_category_new', 'adi_category_new']
        elif model_type == 'simple':
            feature_cols = ['lungrads_12_3_4', 'department_new']
        return feature_cols

    def nb_cv(self, predictors, outcome, kf, model_type=None):
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
        feature_cols = self.get_feature_cols(model_type=model_type)
        # print(feature_cols)

        for train_index, test_index in kf.split(predictors):
            X_train, X_test = predictors.iloc[train_index, :], predictors.iloc[test_index, :]
            y_train, y_test = outcome[train_index], outcome[test_index]
            # add the outcome node (i.e., y_train) to X_train so that the data matrix contains data on all nodes
            X_train['adherence_altered'] = y_train
            # fit model
            nb_model = self.nb_model(model_type=model_type)
            nb_model.fit(X_train, complete_samples_only=False) # set complete_samples_only=False to include observations with missing values
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
                rocauc = roc_auc_score(y_test, prediction_list)

                recall_list.append(recall)
                precision_list.append(precision)
                acc_list.append(acc)
                rocauc_list.append(rocauc)
                print('recall: ', recall)
                print('precision:', precision)
                print('acc: ', acc)
                print('rocauc: ', rocauc)
                print('------------------------')
        return recall_list, precision_list, acc_list, rocauc_list

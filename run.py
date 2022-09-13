from pgmpy.models.BayesianNetwork import BayesianNetwork
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold, KFold
from Experiment_2.utils import read_data, create_var_df, train_data, train_nested, print_bbm_results, build_final_model,\
    test_X_y, test_results, nb_cv

##### Model training #####
#read training data
complete_data_path = '~/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_complete.csv'
missing_data_path = '~/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_missing.csv'
test_data_path = '~/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_test_data.csv'

baseline_data_complete = read_data(complete_data_path)
baseline_data_all = read_data(missing_data_path)

# train and validate models
print('Cross validation')
# complete cases analysis n=1918
var_df_complete = create_var_df(baseline_data_complete)
X, y_nonadherent = train_data(baseline_data_complete, var_df_complete)
train_nested(X, y_nonadherent)

###### naive bayes with missing values #####
kf = RepeatedKFold(n_splits=3, random_state=1, n_repeats=5)
# full model - missing values
nb_full = BayesianNetwork([('adherence_altered', 'lungrads_12_3_4'),
                      ('adherence_altered', 'education_new'),
                      ('adherence_altered', 'median_income_category_new'),
                      ('adherence_altered', 'fam_hx_lc_new'),
                      ('adherence_altered', 'comorbid_category_new'),
                      ('adherence_altered', 'department_new')])
feature_cols = ['lungrads_12_3_4',
       'education_new', 'median_income_category_new',
       'fam_hx_lc_new', 'comorbid_category_new',
       'department_new']
X_missing = baseline_data_all.iloc[:,:-1]
y_missing = baseline_data_all.iloc[:,-1]
recall_list_missing, precision_list_missing, acc_list_mising, rocauc_list_missing = nb_cv(X_missing, y_missing, nb_full, kf, feature_cols)
print_bbm_results(recall_list_missing, precision_list_missing, acc_list_mising, rocauc_list_missing)

# test final full model using LR
print('Testing')
final_full_model = build_final_model(baseline_data_complete, LogisticRegression())
final_full_test_X, final_full_test_y = test_X_y(test_data_path)
test_results(final_full_model, final_full_test_X, final_full_test_y)


from dataloader import DataLoader
from models import train_nested, print_bn_results, test_results
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # data paths
    full_complete_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_complete.csv'
    full_missing_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_missing.csv'
    full_imputed_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_imputed.csv'
    simple_complete_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_simple_complete.csv'
    simple_missing_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_simple_missing.csv'
    simple_imputed_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_simple_imputed.csv'
    test_data_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_test_data.csv'

    # complete and imputed data
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
    # Note: in cv_outer, if using dummy data, change 10 to 3 to avoid getting an error for roc-auc: 'Only one class present in y_true. ROC AUC score is not defined in that case.'
    cv_outer = RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)

    ####################################################################################################################
    # first specify train or test
    # options: 'cross validation', 'test'
    cv_or_test = 'cross validation'

    # then specify data type
    # options: 'full_complete', 'simple_complete', 'full_imputed', 'simple_imputed', 'full_missing', 'simple_missing'
    # Note: for testing final full model, specify 'full_imputed' as the final model was trained on the full complete data
    # Note: for testing final simple model, specify 'simple_complete' as the final model was trained on the simple complete data
    train_data_type = 'simple_missing'
    ####################################################################################################################

    print('------', cv_or_test, '------')
    if cv_or_test == 'test':
        print('------', train_data_type.split('_')[0], '------')
    else:
        print('------', train_data_type, '------')

    # cross validation on complete and imputed data sets
    if cv_or_test == 'cross validation' and train_data_type == 'full_complete':
        X, y = DataLoader(full_complete_path).get_data(model_type='full')
        train_nested(X, y, cv_inner, cv_outer)
    elif cv_or_test == 'cross validation' and train_data_type == 'simple_complete':
        X, y = DataLoader(simple_complete_path).get_data(model_type='simple')
        train_nested(X, y, cv_inner, cv_outer)
    elif cv_or_test == 'cross validation' and train_data_type == 'full_imputed':
        X, y = DataLoader(full_imputed_path).get_data(model_type='full')
        train_nested(X, y, cv_inner, cv_outer)
    elif cv_or_test == 'cross validation' and train_data_type == 'simple_imputed':
        X, y = DataLoader(simple_imputed_path).get_data(model_type='simple')
        train_nested(X, y, cv_inner, cv_outer)

    # cross validation on data with missing values
    if cv_or_test == 'cross validation' and train_data_type == 'full_missing':
        X, y = DataLoader(full_missing_path).get_data_nb()
        print_bn_results(X, y, cv_outer, model_type='full')
    elif cv_or_test == 'cross validation' and train_data_type == 'simple_missing':
        X, y = DataLoader(simple_missing_path).get_data_nb()
        print_bn_results(X, y, cv_outer, model_type='simple')

    # testing final models
    if cv_or_test == 'test' and train_data_type == 'full_imputed':
        test_results(classifier=LogisticRegression(),
                     model_type='full',
                     file_path=full_imputed_path,
                     file_type='full_imputed',
                     test_path=test_data_path)
    elif cv_or_test == 'test' and train_data_type == 'simple_complete':
        test_results(classifier=LogisticRegression(),
                     model_type='simple',
                     file_path=simple_complete_path,
                     file_type='simple_complete',
                     test_path=test_data_path)


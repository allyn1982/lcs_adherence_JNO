import pandas as pd
from utils import create_var_df, create_var_df_simple, train_data

class DataLoader:
    """Data loader for all types of data setes.
    Input:
    file_path: string, path to training data

    Output:
    One-hot encoded predictors (X) and outcome (y) for each data set.
    """
    def __init__(self, file_path):
        self.file_path = file_path

    def get_data(self, model_type=None):
        """Read data for logistic regression, SVM, random forest, and XGBoost classifiers
        """
        data = pd.read_csv(self.file_path)
        # specify predictors for full and simple models
        if model_type == 'full':
            var_df = create_var_df(data)
        elif model_type == 'simple':
            var_df = create_var_df_simple(data)
        # one-hot encoding
        X, y = train_data(data, var_df)
        return X, y

    def get_data_nb(self):
        """Read data for naive bayes classifiers
        """
        data = pd.read_csv(self.file_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        return X, y

    def get_test_data(self,  model_type=None, test_path=None):
        """Read hold out test data
        """
        test_data = pd.read_csv(test_path)
        if model_type == 'full':
            test_var_df = create_var_df(test_data)
        elif model_type == 'simple':
            test_var_df = create_var_df_simple(test_data)
        final_test_X, final_test_y = train_data(test_data, test_var_df)
        return final_test_X, final_test_y

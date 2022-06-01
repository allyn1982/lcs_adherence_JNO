from model_test import Model_test
from sklearn.linear_model import LogisticRegression

class Test:
    """Print results on the hold out test set
    Input:
    classifier: specifier classifier to retrain the model
    model_type: 'full' or 'simple'
    file_path: path to final training data set
    file_type: type of final training data set (i.e., 'full_imputed' or 'simple_complete')
    test_path: path to hold out test data set

    Output:
    Print recall, precision, accuracy, and ROC-AUC for full and simple models.
    """
    def __init__(self, classifier, model_type, file_path, file_type, test_path):
        self.classifier = classifier
        self.model_type = model_type
        self.file_path = file_path
        self.file_type = file_type
        self.test_path = test_path

    def print_test_restuls(self):
        Model_test().test_results( file_path=self.file_path,
                                   file_type=self.file_type,
                                   model_type=self.model_type,
                                   classifier=self.classifier,
                                   test_path=self.test_path)

if __name__ == '__main__':
    test_data_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_test_data.csv'
    full_imputed_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_full_imputed.csv'
    simple_complete_path = 'C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_2/Dummy_Data/Dummy_Data_Experiment_2_simple_complete.csv'

    # full model
    print('Testing full model')
    Test(classifier=LogisticRegression(), model_type='full', file_path=full_imputed_path,
         file_type='full_imputed', test_path=test_data_path).print_test_restuls()

    # simple model
    print('Testing simple model')
    Test(classifier=LogisticRegression(), model_type='simple', file_path=simple_complete_path,
         file_type='simple_complete', test_path=test_data_path).print_test_restuls()



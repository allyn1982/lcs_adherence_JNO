from sklearn.metrics._classification import recall_score, precision_score, accuracy_score
from sklearn.metrics._ranking import roc_auc_score
from dataloader import DataLoader

class Model_test:
    """Retrain final models and test on hold out test set.

    """
    def build_final_model(self, classifier=None, model_type=None, file_path=None, file_type=None):
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
        final_train_X, final_train_y = DataLoader(file_path, file_type=file_type).get_data(model_type=model_type)
        final_model.fit(final_train_X, final_train_y)
        return final_model

    def test_results(self, file_path=None,  file_type=None, model_type=None, classifier=None, test_path=None):
        """Function to get results on test data.
        Input:
        file_path: path to training data
        file_type: 'full_imputed' or 'simple_complete'
        model_type: 'full' or 'simple'
        classifier: specify the classifier
        test_path: path to hold out test data
        
        Output:
        Testing results for each metric.
        """
        final_test_X, final_test_y = DataLoader(file_path=file_path, file_type=file_type).get_test_data(model_type=model_type, test_path=test_path)
        model = self.build_final_model(classifier=classifier, model_type=model_type, file_path=file_path, file_type=file_type)
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

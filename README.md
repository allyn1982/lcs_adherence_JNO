# Identifying predictors of and modeling patient non-adherence to lung cancer screening across multiple screening time points.
*  Authors: Yannan Lin, Li-Jung Liang, Ruiwen Ding, Denise R Aberle, Ashley Elizabeth Prosper, William Hsu
* Affiliation: UCLA

In this article, we conducted three experiments.

* Experiment_1: Identifying predictors of patient non-adherence to baseline Lung-RADS recommendations using logistic regression.

* Experiment_2: Using machine learning-based models to predict patient non-adherent to baseline Lung-RADS recommendations.

* Experiment_3: Investigating the association between patterns in Lung-RADS scores and changes in patient non-adherence to Lung-RADS recommendations across multiple screening time points.

## Notes ##
1. Dummy data is provided for each experiment to help run the model.
2. When running each model, change the directory of the dummy data to your local directory.
3. Experiment 2: When having import errors, check the versions of your python packages using requirements.txt.
4. Experiment 2: When using dummy data sets, set num_splits=3 in utils.py and n_splits=3 in run.py.

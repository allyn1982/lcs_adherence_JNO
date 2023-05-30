# Factors Associated With Nonadherence to Lung Cancer Screening Across Multiple Screening Time Points
# https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2805301?resultClick=1
*  Authors: Yannan Lin, Li-Jung Liang, Ruiwen Ding, Denise R Aberle, Ashley Elizabeth Prosper, William Hsu
* Affiliation: UCLA

In this article, we conducted three experiments.

* Experiment_1 (Experiment 1 in the article): Identifying predictors of patient non-adherence to baseline Lung-RADS recommendations using logistic regression.

* Experiment_2 (Supplement experiment in the article): Using machine learning-based models to predict patient non-adherent to baseline Lung-RADS recommendations.

* Experiment_3 (Experiemnt 2 in the article): Investigating the association between patterns in Lung-RADS scores and changes in patient non-adherence to Lung-RADS recommendations across multiple screening time points.

## Notes ##
1. Dummy data is provided for each experiment to help run the model.
2. When running each model, change the directory of the dummy data to your local directory.
3. Experiment 2: When having import errors, check the versions of your python packages using requirements.txt.
4. Experiment 2: When using dummy data sets, set num_splits=3 in utils.py and n_splits=3 in run.py.

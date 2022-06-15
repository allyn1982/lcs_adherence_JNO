# Identifying and modeling patient non-adherence to lung cancer screening across multiple screening time points.
*  Authors: Yannan Lin, Li-Jung Liang, Ruiwen Ding, Denise R Aberle, Ashley Elizabeth Prosper, William Hsu
* Affiliation: UCLA

In this article, we conducted three experiments.

* Experiment_1: Identifying predictors of patient non-adherence to baseline Lung-RADS recommendations using logistic regression.

* Experiment_2: Using machine learning-based models to predict patient non-adherent to baseline Lung-RADS recommendations.

* Experiment_3: Investigating the association between changes in Lung-RADS scores and patient non-adherence to Lung-RADS recommendations across multiple screening time points.

## Experiment 2 ##
The machine leaning models can be trained and tested using the main.py script.

Command Line examples:
1. Cross validation on the complete data set for the parsimonious model

`python main.py --fold 3 --cv_or_test cross_validation  --train_data_type simple_complete`

2. Test the final full model

`python main.py --cv_or_test test  --train_data_type full_complete
`

3. Test the final parsimonious model

`python main.py --cv_or_test test  --train_data_type simple_imputed
`

## Notes ##
1. Dummy data is provided for each experiment to help run the model.
2. When running each model, change the directory of the dummy data to your local directory.
3. Experiment 2: When having import errors, check the versions of your python packages using requirements.txt.
4. Experiment 2: When using dummy data sets, type `--fold 3` (instead of 10)in command line.

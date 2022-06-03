# experiment 2
# logistic regression with lasso for variable selection
# complete case data

# Baseline predictors
# Lung-RADS score: lungrads_12_3_4
# Age:age_new
# Sex: sex_new
# Race/ethnicity: race_ethnicity_new
# Education: education_new
# Smoking status: smoking_status_new
# Family history of lung cancer: fam_hx_lc_new
# Primary insurance: insurance_new
# Age-adjusted Charlson Comorbidity Index: comorbid_category_new
# Distance to screening center: distance_to_center_category_new
# ADI state rank: adi_category_new
# Type of referring physician: department_new
# Median family income: median_income_category_new
# COVID pause period: covid_expected_fu_date_lungrads_interval_new

# Outcome
# Adherence: adherence_altered

# make sure 'glmnet' is installed
if(!require(glmnet)) install.packages('glmnet', repos = "http://cran.us.r-project.org")

# read data
my_data = read.csv('~/Dummy_Data_Experiment_2_LASSO.csv')
# create a data matrix for predictors
x <- model.matrix(my_data$adherence_altered ~
                    my_data$lungrads_12_3_4+
                    my_data$age_new+
                    my_data$sex_new+
                    my_data$race_ethnicity_new+
                    my_data$education_new+
                    my_data$smoking_status_new+
                    my_data$fam_hx_lc_new+
                    my_data$comorbid_category_new+
                    my_data$department_new +
                    my_data$distance_to_center_category_new+
                    my_data$adi_category_new+
                    my_data$median_income_category_new+
                    my_data$insurance_new+
                    my_data$covid_expected_fu_date_lungrads_interval_new)[, -1]

# # alpha=1 is only lasso, alpha=0 is only ridge, otherwise both l1 and l2 norm
# glmmod <- glmnet(x, y=as.factor(my_data$adherence_altered), alpha = 1, family = 'binomial')
# plot(glmmod, xvar='lambda')
# # coefficients
# coef(glmmod)[,10]

# use cv to select best lambda
set.seed(5)
cv.glmmod <- cv.glmnet(x, y=my_data$adherence_altered, alpha=1, nfolds = 10)
plot(cv.glmmod)
(best.lambda <- cv.glmmod$lambda.1se)
# interpret results with cv - 1se
coef(cv.glmmod, s='lambda.1se')

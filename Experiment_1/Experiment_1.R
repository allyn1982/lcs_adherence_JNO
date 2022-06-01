# experiment 1 
# build a multivariable model to identify significant predictors of patient non-adherence to baseline Lung-RADS recommendations

# Baseline predictors
# Lung-RADS score: lungrads_12_3_4
# Age:age_new
# Sex: sex_new
# Race/ethnicity: race_ethnicity_new
# Education: education_new
# Smoking status: smoking_status_new
# Family history of lung cancer: fam_hx_lc_new
# Primary insurance: insurance_new
# Age-adjusted Charlson Comorbidity Index: comorbid_category_new_2
# Distance to screening center: distance_to_center_category_new
# ADI state rank: adi_category_new
# Type of referring physician: department_new
# Median family income: median_income_category_new
# COVID pause period: covid_expected_fu_date_lungrads_interval_new

# Outcome
# Adherence: adherence_altered

multi_lr_model <- function(my_data){
  # multiviariable logistic regression model
  model <- glm(adherence_altered ~ lungrads_12_3_4+age_new+
                 sex_new+race_ethnicity_new+
                 education_new+smoking_status_new+
                 fam_hx_lc_new+insurance_new+
                 comorbid_category_new_2+distance_to_center_category_new+
                 adi_category_new+department_new+median_income_category_new+
                 covid_expected_fu_date_lungrads_interval_new,
               data = my_data,
               family = binomial(link="logit"))
  
  # number of observations
  print(nobs(model))
  
  # coefficients and its 95% CI
  round(exp(cbind(coef(model), confint(model))), digits = 2)
}

# run logistic regression on complete data - cases with missing values will be excluded automatically
my_data_all <- read.csv('path_to_data')
multi_lr_model(my_data_all)

# experiment 1
multi_lr_model <- function(my_data){
  # full model - multiviariable
  model <- glm(adherence_altered ~ lungrads_12_3_4+age_new+
                 sex_new+race_ethnicity_new+
                 education_new+smoking_status_new+
                 fam_hx_lc_new+insurance_new+
                 comorbid_category_new_2+distance_to_center_category_new+
                 adi_category_new+department_new+median_income_category_new+
                 covid_expected_fu_date_lungrads_interval_new,
               data = my_data,
               family = binomial(link="logit"))
  print(nobs(model))
  round(exp(cbind(coef(model), confint(model))), digits = 2)
}

# run logistic regression on complete data - cases with missing values will be excluded automatically
my_data_all <- read.csv('path_to_data')
multi_lr_model(my_data_all)

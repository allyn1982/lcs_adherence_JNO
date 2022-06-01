# experiment 3
# investigate the association between changes in Lung-RADS scores and patient adherence across mulitple screening time points

# Baseline predictors
# Lung-RADS score: baseline_lr_category
# Age: baseline_age
# Sex: sex_new
# Race/ethnicity: baseline_race_ethnicity
# Education: baseline_education
# Smoking status: baseline_smoking_status
# Family history of lung cancer: baseline_fam_hx_lc
# Primary insurance: baseline_insurance
# Age-adjusted Charlson Comorbidity Index: baseline_comorbid_category
# Distance to screening center: baseline_distance_to_center_category
# ADI state rank: baseline_adi_category
# Type of referring physician: baseline_department
# Median family income: baseline_median_income_category
# COVID pause period: covid_expected_fu_date_lungrads_interval_new
# Screening time point: ldct_index
# Grouping for changes in Lung-RADS score over time: lr_change_modified_new

# Outcome
# Adherence: adherence_altered

library(gee)

# extract 95% CI for coefficients for variables of interest only
get_CI <- function(model){
  print('Baseline LR 1-2, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['ldct_index2',4]))
  print('Baseline LR 1-2, LR unchanged from T0 to T2, T0 vs T2')    
  print(exp(coef(summary(model))['ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['ldct_index3',4]))
  print('Baseline LR 3-4, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3-4:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3-4:ldct_index2',4]))
  print('Baseline LR 3-4, LR unchanged from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3-4:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3-4:ldct_index3',4]))
  print('Baseline LR 1-2, LR upgraded  from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['lr_change_modified_newchange:ldct_index2',4]))
  print('Baseline LR 1-2, LR upgraded  from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['lr_change_modified_newchange:ldct_index3',4]))
  print('Baseline LR 3-4, LR downgraded from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index2',4]))
  print('Baseline LR 3-4, LR downgraded from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index3',4]))
}

# extract odds ratios for variables of interest only
get_OR <- function(model){
  print('Baseline LR 1-2, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['ldct_index2',1]))
  print('Baseline LR 1-2, LR unchanged from T0 to T2, T0 vs T2')   
  print(exp(coef(summary(model))['ldct_index3',1]))
  print('Baseline LR 3-4, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3-4:ldct_index2',1]))
  print('Baseline LR 3-4, LR unchanged from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3-4:ldct_index3',1]))
  print('Baseline LR 1-2, LR upgraded  from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index2',1]))
  print('Baseline LR 1-2, LR upgraded  from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index3',1]))
  print('Baseline LR 3-4, LR downgraded from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index2',1]))
  print('Baseline LR 3-4, LR downgraded from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3-4:lr_change_modified_newchange:ldct_index3',1]))
}

# build gee model
gee_analysis <- function(data){
  # adjusted model
  model_complete_adjusted <- gee(adherence_altered ~
                                   baseline_lr_category*lr_change_modified_new*ldct_index +
                                   baseline_age + sex_new +
                                   baseline_race_ethnicity+baseline_education+
                                   baseline_smoking_status+ baseline_fam_hx_lc+
                                   baseline_comorbid_category+baseline_insurance+
                                   baseline_department+ baseline_distance_to_center_category +
                                   baseline_adi_category+baseline_median_income_category+
                                   covid_expected_fu_date_lungrads_interval_new,
                                 data = data,
                                 id = mrn,
                                 family = binomial,
                                 maxiter = 100,
                                 corstr = "unstructured")
  print(2 * pnorm(abs(coef(summary(model_complete_adjusted))[,5]), lower.tail = FALSE))
  # get OR and 95% CI
  get_OR(model_complete_adjusted)
  get_CI(model_complete_adjusted)
}

# primary analysis - without imputation
no_imputation_data = read.csv("path_to_file")
gee_analysis(no_imputation_data)

# sensitivity analysis - with imputation
imputed_data = read.csv("path_to_file")
gee_analysis(imputed_data)



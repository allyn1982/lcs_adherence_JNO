# experiment 3

library(gee)

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



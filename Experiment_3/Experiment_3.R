# experiment 3
# investigate the association between changes in Lung-RADS scores and patient adherence across mulitple screening time points

# Baseline predictors
# id: research_id
# Lung-RADS score: baseline_lr_category
# Education: baseline_education
# Family history of lung cancer: baseline_fam_hx_lc
# Age-adjusted Charlson Comorbidity Index: baseline_comorbid_category
# Type of referring physician: baseline_department
# Median family income: baseline_median_income_category
# Screening time point: ldct_index
# Grouping for changes in Lung-RADS score over time: lr_change_modified_new

# Outcome
# Adherence: adherence_altered

library(gee)

# extract 95% CI for coefficients for variables of interest only
get_CI <- function(model){
  print('Baseline LR 1to2, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['ldct_index2',4]))
  print('Baseline LR 1to2, LR unchanged from T0 to T2, T0 vs T2')    
  print(exp(coef(summary(model))['ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['ldct_index3',4]))
  print('Baseline LR 3to4, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3to4:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3to4:ldct_index2',4]))
  print('Baseline LR 3to4, LR unchanged from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3to4:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3to4:ldct_index3',4]))
  print('Baseline LR 1to2, LR upgraded  from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['lr_change_modified_newchange:ldct_index2',4]))
  print('Baseline LR 1to2, LR upgraded  from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['lr_change_modified_newchange:ldct_index3',4]))
  print('Baseline LR 3to4, LR downgraded from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index2',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index2',4]))
  print('Baseline LR 3to4, LR downgraded from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index3',1] + 
              qnorm(c(0.025, 0.975)) * coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index3',4]))
}

# extract odds ratios for variables of interest only
get_OR <- function(model){
  print('Baseline LR 1to2, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['ldct_index2',1]))
  print('Baseline LR 1to2, LR unchanged from T0 to T2, T0 vs T2')   
  print(exp(coef(summary(model))['ldct_index3',1]))
  print('Baseline LR 3to4, LR unchanged from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3to4:ldct_index2',1]))
  print('Baseline LR 3to4, LR unchanged from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3to4:ldct_index3',1]))
  print('Baseline LR 1to2, LR upgraded  from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index2',1]))
  print('Baseline LR 1to2, LR upgraded  from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['lr_change_modified_newchange:ldct_index3',1]))
  print('Baseline LR 3to4, LR downgraded from T0 to T1, T0 vs T1')
  print(exp(coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index2',1]))
  print('Baseline LR 3to4, LR downgraded from T0 to T2, T0 vs T2')
  print(exp(coef(summary(model))['baseline_lr_category3to4:lr_change_modified_newchange:ldct_index3',1]))
}

# build gee model
gee_analysis <- function(data){
  # set ldct index as factor
  data$ldct_index <- as.factor(data$ldct_index)
  # reorder levels
  data <- within(data, lr_change_modified_new <- relevel(as.factor(lr_change_modified_new), ref = 2))
  levels(data$lr_change_modified_new)
  
  # adjusted model
  model_complete_adjusted <- gee(adherence_altered ~
                                   baseline_lr_category*lr_change_modified_new*ldct_index +
                                   baseline_education+
                                   baseline_fam_hx_lc+
                                   baseline_comorbid_category+
                                   baseline_department+ 
                                   baseline_median_income_category,
                                 data = data,
                                 id = research_id,
                                 family = binomial,
                                 maxiter = 100,
                                 corstr = "unstructured")
  print(2 * pnorm(abs(coef(summary(model_complete_adjusted))[,5]), lower.tail = FALSE))
  # get OR and 95% CI
  print('OR')
  get_OR(model_complete_adjusted)
  print('95% CI')
  get_CI(model_complete_adjusted)
}

# primary analysis - without imputation
no_imputation_data = read.csv("C:/Users/yannanlin/Desktop/Paper 3/code/lcs_adherence/Experiment_3/Dummy_Data_Experiment_3.csv")
gee_analysis(no_imputation_data)



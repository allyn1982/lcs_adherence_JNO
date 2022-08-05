from sklearn.preprocessing import LabelEncoder

def create_var_df(data):
    """Function to select predictors for full models
    Input: a Pandas data frame
    Output: a data frame including only predictors for full models
    """
    return data[['age_new', 'sex_new', 'race_ethnicity_new',
                 'smoking_status_new', 'education_new',
                 'fam_hx_lc_new', 'comorbid_category_new', 'lungrads_12_3_4',
                 'insurance_new', 'distance_to_center_category_new',
                 'department_new', 'adi_category_new', 'median_income_category_new',
                 'covid_expected_fu_date_lungrads_interval_new']]


def create_var_df_simple(data):
    """Function to select predictors for simple models
    Input: a Pandas data frame
    Output: a data frame including only predictors for simple models
    """
    return data[['lungrads_12_3_4', 'department_new']]


def train_data(data, var_df):
    """Function to one-hot/label encode all non-numeric cols
    Input:
    data: a Pandas data frame
    var_df: a data frame including predictors

    Output:
    var_df: a One-hot/label encoded data frame of predictors
    y: a 1-d numpy array
    """
    var_list = list(var_df.columns)
    enc = LabelEncoder()
    # transform x
    for var in var_list:
        if var == 'race_ethnicity_new' or var == 'insurance_new':
            one_hot = pd.get_dummies(var_df[var])
            # Drop column as it is now encoded
            var_df = var_df.drop(var, axis=1)
            # Join the encoded df
            var_df = var_df.join(one_hot)
        else:
            var_df[var] = enc.fit_transform(var_df[var])
    y = data[['adherence_altered']].values.flatten()
    return var_df, y

def get_var_dict(feature_cols, test_case):
    """Function to get a dictionary with keys being feature_cols and values being test_case.
    Input:
    feature_cols: names of variables to be included in the model
    test_case: values of predictors for each test case

    Output:
    evidence_dict: a dictionary with keys being variable names
    and values being the ground truth value for that feature
    """
    evidence_dict = {}
    for i in range(len(feature_cols)):
        feature_name = feature_cols[i]
        feature_value = test_case[i]
        if str(feature_value) != 'nan':
            evidence_dict[feature_name] = feature_value
    return evidence_dict

"""
=====================================================================
Pipeline Summary
=====================================================================

This script performs survival modeling using XGBoost with Cox regression
on 10 MICE-imputed datasets of ICU COVID-19 patients.

Pipeline overview:
------------------
1. **Data Import**
   - Loads 10 MICE-imputed Excel datasets into a dictionary.
   - Loads the original dataset with missing values to define which
     patients are used for internal validation.

2. **Model Setup**
   - Loads pre-optimized XGBoost hyperparameters.
   - Configures model for survival analysis using Cox objective.

3. **Cross-Validation**
   - Uses 5-fold cross-validation repeated 20 times (100 total splits).
   - Trains and evaluates model on each of the 10 imputed datasets.

4. **Explainability**
   - Computes SHAP values to explain individual predictions.
   - Averages SHAP values and feature inputs across imputations.

5. **Evaluation**
   - Calculates Concordance Index (C-index) for train/validation.
   - Bootstraps 50 times to produce 95% confidence intervals.

6. **Output**
   - Saves predictions, SHAP values, averaged features, and metrics to Excel.
   - Writes C-index summaries to text files.

=====================================================================
"""

# -------------------------------------------------------------------
# Import required packages and setup environment
# -------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
from optuna import create_study
from sksurv.metrics import concordance_index_censored

from collections import deque
import torch
from xgboost import XGBRegressor
import scipy.stats as st
from typing import Any, Callable
from org import excel, write_to_file # local scripts for organising output
from sklearn.model_selection import RepeatedStratifiedKFold
from datetime import datetime

# -------------------------------------------------------------------
# Global parameters and constants
# -------------------------------------------------------------------
date_time = datetime.today().strftime('%Y-%m-%d-%H:%M:%S') # Date and time to save output
REPETITIONS = 20
FOLDS = 5
SEED = 42
nbr_of_trials = 1 # Since we are not doing a trial but running the pipeline once
RESAMPLE = 10
n_bootstraps = 50
NAME = 'survival_analysis_prescriptions_10_MICE'
rng = np.random.default_rng(seed=SEED)

# Mapping of internal variable names to human-readable names for plots
change_names = {'clinical_frailty_scale': 'CFS', 'leukocytes': 'Leukocytes', 'albumin': 'Albumin', 'ngal': 'NGAL',
                'ddimer': 'D-dimer', 'il6': 'IL-6', 'age_at_admission': 'Age', 'meds_per_pat': '#Meds/Patient',
                'time_at_hospital_before_admission': '#Days in Hosp. before ICU', 'no_CPR': 'DNR order',
                'thrombocytes': 'WBC', 'neutrofila': 'Neutrophils', 'A02BC': 'PPI medication',
                'cystatin_c': 'Cystatin C',
                'smoker': 'Smoker',
                'congestive_heart_failure': 'Congestive heart failure',
                'cancer': 'Cancer',
                'hypertension': 'Hypertension',
                'myocardial_infarction': 'Myocardial infarction',
                'peripheral_vascular_disease': 'Peripheral vascular disease',
                'cerebrovascular_insult': 'Cerebrovascular insult',
                'chronic_pulmonary_disease': 'Chronic pulmonary disease',
                'rheumatic_disease': 'Rheumatic disease',
                'peptic_ulcer_disease': 'Peptic ulcer disease',
                'diabetes_mellitus_uncompl': 'Diabetes mellitus (uncompl.)',
                'diabetes_mellitus_compl': 'Diabetes mellitus (compl.)',
                'chronic_kidney_disease': 'Chronic kidney disease',
                'imv_within_24h': 'IMV within 24h from ICU admission',
                'noradrenaline': 'Noradrenaline',
                'sex_male': 'Sex',
                'GCS': 'Glasgow Coma Scale',
                'cardiovascular_map': 'Cardiovascular mean arterial pressure',
                'creatinine': 'Creatinine',
                'bilirubin': 'Bilirubin',
                'ferritin': 'Ferritin',
                'laktat': 'Lactate',
                'pct': 'Procalcitonin',
                'pao2': 'Pa0$_2$',
                'paco2': 'PaC0$_2$',
                'aB_pH': 'pH (arterial blood)',
                'lymphocytes': 'Lymphocytes',
                'body_temperature': 'Body temperature (\N{DEGREE SIGN}C)',
                'heart_frequency': 'Heart frequency (Hz)',
                'systolic_blood_pressure': 'Systolic blood pressure (mmHg)',
                'arterial_oxygen_tension': 'Arterial oxygen tension (mmHg)',
                'symptomatic_days': '#Symptomatic days',
                'habitual_creatinine': 'Habitual creatinine',
                'ICU_strain': 'ICU burden',
                'endostatin': 'Endostatin',
                'vcam': 'VCAM',
                'icam': 'ICAM',
                'calprotectin': 'Calprotectin',
                'pf_quotient': 'PaC0$_2$/FiO$_2$ Ratio',
                'survival_days_from_icu_start': '#Survival days from ICU admission',
                'mort': 'Dead within one year',
                'study_month': 'Study month'
                }  # 'NOAK', 'CRP', 'BMI', 'CCI', 'ACEi', 'ARB'

# -------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------
def plot_names(variables_to_rename, feat_name, feat_names):
    """
    Replace technical variable names with human-readable names for plotting.

    Args:
        variables_to_rename (dict): Mapping of old → new feature names.
        feat_name (str): A single feature name to rename.
        feat_names (list[str]): List of feature names to check and rename.

    Returns:
        tuple: Updated (feat_name, feat_names) with renamed variables.
    """
    for old, new in variables_to_rename.items():
        if feat_name == old:
            feat_name = new
        elif old in feat_names:
            feat_names = [new if x == old else x for x in feat_names]
    return feat_name, feat_names


# -------------------------------------------------------------------
# Data Import and Preparation
# -------------------------------------------------------------------
"""
This section loads the 10 MICE-imputed Excel datasets and the version
with missing data (used to identify internal validation full-case patients).
Each imputed dataset is stored in a dictionary indexed by dataset number.
"""
n_files = 10  # Number of imputed datasets

# Create a list to store DataFrames
c19_survival_dict = {}
path_to_folder = Path(r"path_to_imputed_data")
# Loop to read each Excel file
for i in range(n_files):
    filename = os.path.join(path_to_folder, f'imputed_values_{i}.xlsx')
    df = pd.read_excel(filename, engine='openpyxl')   # Read Excel file into a DataFrame
    df = df.drop('Unnamed: 0', axis=1)
    c19_survival_dict[i] = df             # Add to dict

# Load dataset containing missing values (pre-imputation)
c19_survival_missing = pd.read_excel(r"path_to_original_data", engine='openpyxl')

# Identify patients with missing data to identify which patients should be in internal validation dataset
missing_idx = c19_survival_missing[c19_survival_missing.isna().any(axis=1)].index

# Print dataset checks and summary statistics
print('Variables:', c19_survival_dict[0].columns.to_list()) # Check for one dataset

# Check which (binary or ordinal) variables looses some values when omitting non full-case
full_data = [idx for idx in c19_survival_dict[0].index if not idx in missing_idx]
full_cases = c19_survival_dict[0].loc[c19_survival_dict[0].index.isin(full_data)]
print('Patients with full-case data: ', str(len(full_cases)), '/', str(len(c19_survival_dict[0])), '(',
      str(np.round(len(full_cases) / len(c19_survival_dict[0]), 2) * 100), '%)')
print('Complete variables:', str(len(c19_survival_missing.dropna(axis=1).columns)), '/',
      len(c19_survival_missing.columns), '(', str(np.round(
        len(c19_survival_missing.dropna(axis=1).columns) / (len(c19_survival_missing.columns)),
        2) * 100), '%)')

print(full_cases['CCI'].unique().argsort().tolist())  # [5, 3, 4, 2, 7, 0, 1, 6, 8, 9, 11, 10]

# -------------------------------------------------------------------
# SHAP label dictionary for consistent plotting and naming
# -------------------------------------------------------------------
labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}

# -------------------------------------------------------------------
# SHAP and Model Helper Functions
# -------------------------------------------------------------------
def get_shap_values(
        estimator: Any,
        test_x: pd.DataFrame | np.ndarray | torch.Tensor,
        attribute_names: list[str] | None = None,
        **kwargs: Any,
) -> np.ndarray:
    """Compute SHAP values for a model's predictions on input features.

    This function calculates SHAP (SHapley Additive exPlanations) values that
    attribute the contribution of each input feature to the model's output.
    It automatically selects the appropriate SHAP explainer based on the model.

    Args:
        estimator: The model to explain, typically a TabPFNClassifier or scikit-learn compatible model.
        test_x: The input features to compute SHAP values for.
        attribute_names: Column names for the features when test_x is a numpy array.
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        np.ndarray: The computed SHAP values with shape (n_samples, n_features).
    """
    if isinstance(test_x, torch.Tensor):
        test_x = test_x.cpu().numpy()

    if isinstance(test_x, np.ndarray):
        test_x = pd.DataFrame(test_x)
        if attribute_names is not None:
            test_x.columns = attribute_names
        else:
            test_x = test_x.rename(columns={c: str(c) for c in test_x.columns})

    if hasattr(estimator, "predict_function_for_shap"):
        predict_function_for_shap = estimator.predict_function_for_shap
    else:
        predict_function_for_shap = (
            "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
        )

    if hasattr(estimator, "fit_at_predict_time"):
        if not estimator.fit_at_predict_time:
            pass

    def get_shap() -> np.ndarray:
        explainer = get_default_explainer(
            estimator,
            test_x,
            predict_function_for_shap,
            **kwargs,
        )
        return explainer(test_x)

    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            shap_values = get_shap()
        finally:
            estimator.show_progress = show_progress_
    else:
        shap_values = get_shap()

    return shap_values


def get_default_explainer(
        estimator: Any,
        test_x: pd.DataFrame,
        predict_function_for_shap: str | Callable = "predict",
        **kwargs: Any,
) -> Any:
    """Create a standard SHAP explainer for non-TabPFN models.

    Args:
        estimator: The model to explain.
        test_x: The input features to compute SHAP values for.
        predict_function_for_shap: Function name or callable to use for prediction.
            Defaults to "predict".
        **kwargs: Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        Any: A configured SHAP explainer for the model.
    """
    import shap

    shap.maskers.Independent(test_x, max_samples=1000)

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap)
        if isinstance(predict_function_for_shap, str)
        else predict_function_for_shap,
        test_x,
        **kwargs,
    )


# -------------------------------------------------------------------
# Model Initialization and Hyperparameters
# -------------------------------------------------------------------
def best_params_xgboost() -> XGBRegressor:
    """
        Load and return an XGBoost regressor configured with pre-optimized hyperparameters.

        The hyperparameters are read from an Excel file containing the best parameters
        found during prior hyperparameter optimization. The model uses a Cox survival
        objective for time-to-event prediction.

        Returns:
            XGBRegressor: XGBoost model configured for survival analysis.
        """
    best_params = pd.read_excel(
        r"path_to_optimized_parameters",
        engine='openpyxl')
    best_params = best_params.set_index('Unnamed: 0')

    params = {
        'max_depth': int(best_params.loc['max_depth'].values[0]),
        'learning_rate': best_params.loc['learning_rate'].values[0],
        'subsample': best_params.loc['subsample'].values[0],
        'min_split_loss': best_params.loc['min_split_loss'].values[0],
        'min_child_weight': best_params.loc['min_child_weight'].values[0],
        'reg_lambda': best_params.loc['reg_lambda'].values[0],  # L2
        'reg_alpha': best_params.loc['reg_alpha'].values[0],  # L1
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'tree_method': 'hist'
    }

    print(params)
    return XGBRegressor(**params)

# -------------------------------------------------------------------
# Objective Function for Training and Evaluation
# -------------------------------------------------------------------
def objective(data_deque, non_full_index) -> float:
    """
        Objective function for model development and model evaluation.

        Trains and evaluates the XGBoost Cox model across multiple imputations and
        repeated stratified folds, computes predictions, SHAP values, and performance metrics.

        Workflow:
            1. Loop through 10 imputed datasets.
            2. For each dataset, perform 5x20 repeated stratified CV.
            3. Train XGBoost model and compute predictions.
            4. Compute SHAP values for model explainability.
            5. Aggregate mean predictions and SHAP values across folds/imputations.
            6. Compute train/validation C-index and bootstrap confidence intervals.
            7. Save all outputs to Excel and text files.

        Args:
            data_deque (dict[int, pd.DataFrame]): Dictionary of imputed datasets.
            non_full_index (pd.Index): Indices of patients with missing data (excluded from validation).

        Returns:
            float: Mean validation C-index across imputations.
        """
    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPETITIONS,
                                   random_state=SEED)  # Repeats Stratified K-Fold n times with different randomization in each repetition.

    number_of_patients = len(data_deque[0])
    # Prepare dicts to save stuff for CI and so on
    train_predictions = {key: [] for key in range(number_of_patients)}
    val_predictions = {key: [] for key in range(number_of_patients)}

    shap_values = {key: deque() for key in range(number_of_patients)}
    feature_values = {key: deque() for key in range(number_of_patients)}
    base_values = {key: [] for key in range(number_of_patients)}
    feature_names = []

    ## 1) First loop
    # Get the dataset out of a deque
    for i_loop, data in data_deque.items():
        # Drop unnecessary column
        if 'index' in data.columns.to_list():
            data = data.drop('index', axis=1)

        # Divide into x and y
        XX = data.drop(['mort'], axis=1)
        yy = data.copy(deep=True)

        for cat in data.columns.to_list():
            if cat not in ['mort']:
                yy = yy.drop(cat, axis=1)

        # Instantiate new model for this dataset
        model = best_params_xgboost()  # instantiate_xgboost(trial) #

        ## 2) Second loop: Fold*repetition loop
        for i, (train_index, test_index) in enumerate(rskf.split(XX, yy)):
            X_train = XX.iloc[train_index]
            y_train = yy.iloc[train_index]
            X_test = XX.iloc[test_index]

            # Both time and mortality in y
            y_train.loc[:, 'survival_days_from_icu_start'] = X_train['survival_days_from_icu_start']
            X_train = X_train.drop('survival_days_from_icu_start', axis=1)
            X_test = X_test.drop('survival_days_from_icu_start', axis=1)

            # fit model
            model.fit(X_train, y_train['survival_days_from_icu_start'], sample_weight=y_train['mort'], verbose=True)

            # Calculate SHAP values for each validation dataset
            explanation = get_shap_values(model, X_test,
                                          X_test.columns.to_list())

            # predict on train and test data
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_test)

            # save training predictions from this fold
            for nbr, idx in enumerate(train_index):
                train_predictions[idx].append(train_pred[nbr])

            # save validation predictions from this fold
            for nbr, idx in enumerate(test_index):
                if idx in non_full_index:
                    continue  # skip adding this prediction if it is not a full-case patient
                else:
                    val_predictions[idx].append(val_pred[nbr])
                    # Store validation SHAP values
                    shap_values[idx].append(explanation[nbr].values)
                    feature_values[idx].append(explanation[nbr].data)
                    base_values[idx].append(explanation[nbr].base_values)

            if not len(feature_names) is None:
                feature_names = explanation.feature_names

        # End of loop 2 (folds*repetitions)
    # End of loop 1 (imputed datasets)

    # average predictions over all repeated folds
    train_predictions_mean = {}
    val_predictions_mean = {}
    for idx, predictions in train_predictions.items():
        mean_predictions = np.mean(predictions)
        train_predictions_mean[idx] = mean_predictions
    for idx, predictions in val_predictions.items():
        mean_predictions = np.mean(predictions)
        val_predictions_mean[idx] = mean_predictions

    # Average SHAP over validation results
    shap_mean = {}
    feature_mean = {} # also over the differently imputed datasets
    base_mean = {}

    # SHAP values
    for idx, predictions in shap_values.items():
        mean_predictions = np.mean(predictions, axis=0)  # element-wise over each matrix in each deque
        shap_mean[idx] = mean_predictions
    # Data values
    for idx, predictions in feature_values.items():
        mean_predictions = np.mean(predictions, axis=0)  # element-wise over each matrix in each deque
        feature_mean[idx] = mean_predictions
    # Base values
    for idx, predictions in base_values.items():
        mean_predictions = np.mean(predictions)
        base_mean[idx] = mean_predictions

    # from dict to df
    train_predictions_mean_df = pd.DataFrame.from_dict(train_predictions_mean, orient='index')
    val_predictions_mean_df = pd.DataFrame.from_dict(val_predictions_mean, orient='index')

    # combine with y_true into one df
    predictions_and_targets = train_predictions_mean_df.copy(deep=True)
    predictions_and_targets.rename(columns={0: 'training_predictions'}, inplace=True)
    predictions_and_targets.insert(1, 'validation_predictions', val_predictions_mean_df)
    true_values = XX.copy(deep=True)
    for cat in XX.columns.to_list():
        if cat not in ['survival_days_from_icu_start']:
            true_values = true_values.drop(cat, axis=1)
    true_mort = yy.copy(deep=True)
    for cat in yy.columns.to_list():
        if cat not in ['mort']:
            true_mort = true_mort.drop(cat, axis=1)
    predictions_and_targets.insert(2, 'survival_days_from_icu_start', true_values[
        'survival_days_from_icu_start'])  # since only one var can be in the split, this is taken from X and not y
    predictions_and_targets.insert(3, 'mort', true_mort['mort'])

    # Drop imputed patients that were just used for training
    predictions_and_targets = predictions_and_targets.dropna(axis=0)

    excel(date_time, predictions_and_targets, NAME)

    # c-index, nbr of concordant pairs, nbr of discordant pairs, nbr of pairs with tied risk scores, nbr of comparable pairs with tied survival times
    train_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets['mort'].astype(bool),
                                                          predictions_and_targets['survival_days_from_icu_start'],
                                                          predictions_and_targets['training_predictions'])

    val_cindex, _, _, _, _ = concordance_index_censored(predictions_and_targets['mort'].astype(bool),
                                                        predictions_and_targets['survival_days_from_icu_start'],
                                                        predictions_and_targets['validation_predictions'])
    print(train_cindex, val_cindex)

    # CI for C-index
    train_c = []
    val_c = []
    for re in range(RESAMPLE):
        bs_sample = predictions_and_targets.sample(n=len(predictions_and_targets),
                                                   replace=True) # Sampling with replacement
        cindex_train = concordance_index_censored(bs_sample['mort'].astype(bool),
                                                  bs_sample['survival_days_from_icu_start'],
                                                  bs_sample['training_predictions'])
        cindex_val = concordance_index_censored(bs_sample['mort'].astype(bool),
                                                bs_sample['survival_days_from_icu_start'],
                                                bs_sample['validation_predictions'])
        train_c.append(cindex_train[0])
        val_c.append(cindex_val[0])

    ci_train = st.t.interval(0.95, len(train_c) - 1, loc=train_cindex, scale=st.sem(train_c))
    ci_val = st.t.interval(0.95, len(val_c) - 1, loc=val_cindex, scale=st.sem(val_c))

    print(ci_train)
    print(ci_val)

    text = '\nC-index val: ' + str(round(val_cindex, 3)) + ' (' + str(round(ci_val[0], 3)) + '-' + str(
        round(ci_val[1], 3)) + ')'
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')
    text = '\nC-index train: ' + str(round(train_cindex, 3)) + ' (' + str(round(ci_train[0], 3)) + '-' + str(
        round(ci_train[1], 3)) + ')' + '\n\n'
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')

    # Remove nans
    full_val_shap_mean = {k: v for k, v in shap_mean.items() if not isinstance(v, float)}
    # From dict to df for shap values
    shap_mean_df = pd.DataFrame.from_dict(full_val_shap_mean, orient='index', columns=feature_names)

    # Remove empty deques
    full_val_shap_values = {k: v for k, v in shap_values.items() if
                            bool(v)}  # bool(v) is True if the deque is not empty
    full_val_feature_values = {k: v for k, v in feature_values.items() if
                               bool(v)}  # bool(v) is True if the deque is not empty

    # get all the SHAP values and data from all folds and repetitions into one column per variable
    stored_shap_values = {key: [] for key in range(len(feature_names))}
    stored_feature_values = {key: [] for key in range(len(feature_names))}
    for variable_nbr in range(len(feature_names)):
        print('Variable number', variable_nbr)
        # For SHAP
        for dict_idx, patient_deque in full_val_shap_values.items():
            print('Deque', dict_idx)
            for iteration in range(len(patient_deque)):
                print('Array', iteration)
                stored_shap_values[variable_nbr].append(patient_deque[iteration][variable_nbr])
        # For data
        for dict_idx, patient_deque in full_val_feature_values.items():
            print('Deque', dict_idx)
            for iteration in range(len(patient_deque)):
                print('Array', iteration)
                stored_feature_values[variable_nbr].append(patient_deque[iteration][variable_nbr])

    # from dict to df
    stored_shap_values_df = pd.DataFrame.from_dict(stored_shap_values, orient='columns')
    stored_feature_values_df = pd.DataFrame.from_dict(stored_feature_values, orient='columns')

    # rename columns
    stored_shap_values_df = stored_shap_values_df.set_axis(feature_names, axis=1)
    stored_feature_values_df = stored_feature_values_df.set_axis(feature_names, axis=1)

    ### SAVE for later ###
    # Save shap values and data over all folds
    excel(date_time, stored_shap_values_df, str(NAME) + '_all_shap_values', subfolder=NAME)
    excel(date_time, stored_feature_values_df, str(NAME) + '_all_fold_data', subfolder=NAME)

    # Save the mean of all rounds of SHAP values
    excel(date_time, shap_mean_df, str(NAME) + '_mean_shap_values', subfolder=NAME)

    # Save data (a mean over the 10 imputed datasets)
    # Remove nans
    full_val_feature_mean = {k: v for k, v in feature_mean.items() if not isinstance(v, float)}
    # From dict to df for shap values
    feature_mean_df = pd.DataFrame.from_dict(full_val_feature_mean, orient='index', columns=feature_names)
    excel(date_time, feature_mean_df, str(NAME) + '_features_val_mean', subfolder=NAME)

    return val_cindex

# -------------------------------------------------------------------
# Optuna Study Runner
# -------------------------------------------------------------------
def xgb_train(data):
    """
        Run an Optuna study to optimize or evaluate the XGBoost survival model.

        The study uses the `objective()` function as its optimization target.
        In this implementation, it performs only one trial to execute the full
        pipeline once rather than tuning hyperparameters.

        Args:
            data (dict[int, pd.DataFrame]): Dictionary containing 10 imputed datasets.

        Outputs:
            - Excel files containing trial results and best parameters.
            - Text files summarizing best C-index and used features.
        """
    # A study corresponds to an optimization task, i.e., a set of trials.
    study = create_study(study_name=str(NAME) + '_optimisation', direction='maximize')  # want to maximize the validation C-index
    study.optimize(lambda trial: objective(data, missing_idx),
                   n_trials=nbr_of_trials)

    # Save study parameters and results
    trials_df = study.trials_dataframe()
    if 'value' in trials_df.columns:
        trials_df.rename(columns={'value': 'validation_cindex'}, inplace=True)
    if 'number' in trials_df.columns:
        trials_df.rename(columns={'number': 'trial_number'}, inplace=True)
    excel(date_time, trials_df, 'trials', str(NAME) + '_hp_opt')

    best_params = study.best_params
    best_params_df = pd.DataFrame.from_dict(best_params, orient='index')
    excel(date_time, best_params_df, 'best_params', str(NAME) + '_hp_opt')

    # Save AUC and features used
    best_value = study.best_value
    text = 'C-index: ' + str(round(best_value, 3)) + '\nfeatures: ' + str(data[0].columns.tolist())
    write_to_file(date_time, str(NAME), text, read_text=True, subfolder=str(NAME) + '_hp_opt')

# -------------------------------------------------------------------
# Run the pipeline
# -------------------------------------------------------------------
"""
The final step: executes the full training and evaluation pipeline.
This triggers loading of imputed datasets, model training, SHAP computation,
C-index evaluation, and result export.
"""
xgb_train(c19_survival_dict)


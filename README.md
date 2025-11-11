# Polypharmacy and Proton Pump Inhibitor Use Independently Predict One-Year Mortality in Critical COVID-19: An Explainable AI–Based Survival Analysis
One-year mortality risk ML survival analysis and explainable AI in patients with critical covid-19.<br>
<br>
Preprint available at: https://www.medrxiv.org/content/10.1101/2025.10.27.25338863v1


- Hyperparameter optimization with model development and evaluation
- Explainable AI customized for extracting SHAP values ([Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)) for interval validation data during repeated cross-validation:

 
  The code for preprocessing the data is not available due to the sensitivity of the data. The overall code documentation was generated using a LLM.

## Table of Contents
- [Usage](#usage)
- [Example Plots](#example-plots)
- [Contact](#contact)

---

## Usage
<code>c19_survival_training.py</code> <br>
1. This script builds and evaluates a survival analysis model using XGBoost with a Cox proportional hazards objective to predict mortality among ICU COVID-19 patients.
2. It imports 10 datasets imputed using the MICE algorithm, performs repeated 5-fold cross-validation (20 repetitions, 100 splits), and averages predictions across all imputations.
3. The model’s explainability is handled with SHAP values, which estimate each feature’s contribution to individual survival predictions and are averaged over imputations and folds.
4. Performance is measured using the Concordance Index (C-index), with bootstrap resampling (50 times) to compute 95% confidence intervals for both training and validation results.
5. Finally, the script exports predictions, SHAP explanations, feature data, and evaluation metrics to Excel and text files, providing a complete record of the model’s performance and interpretability.

<code>c19_survival_explainability.py</code> <br>
  - Beeswarm plot for the most important features for one-year mortality prediction
  - Scatter plots with a LOESS curve and a 95% confidence interval
  - Clinically important reference values are highlighted on the LOESS curve when applicable
  - SHAP hazard ratios with 95% confidence interval showing the feature importance for a range of feature values

<code>c19_survival_blobbogram.R</code> <br>
- SHAP values, CIs, and p-values for top ranked clinical predictors, as well as hypertensive medication ACEi and ARB, are visualised in a forest plot
---


## Example plots

## Contact

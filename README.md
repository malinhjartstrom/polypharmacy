# Polypharmacy and Proton Pump Inhibitor Use Independently Predict One-Year Mortality in Critical COVID-19: An Explainable AI–Based Survival Analysis
One-year mortality risk ML survival analysis and explainable AI in patients with critical covid-19.<br>
<br>
Preprint available at: https://www.medrxiv.org/content/10.1101/2025.10.27.25338863v1
<br>
<br>
## Added value
<mark>Explainable AI customized for extracting SHAP values ([Lundberg & Lee, 2017](https://arxiv.org/abs/1705.07874)) for interval validation data during repeated cross-validation and plotting a beeswarm and a dependency plot. </mark>

---

## Scripts

<code>c19_survival_table1.R</code> <br>
- Descriptive statistics for the included patients with critical COVID-19 (n=497)

<code>c19_survival_training.py</code> <br>
- XGBoost with a Cox proportional hazards objective to predict mortality among ICU COVID-19 patients using machine-learning survival analysis
- 10 datasets have been imputed using the MICE algorithm. Five-fold cross-validation (20 repetitions, 100 splits) is performed over all imputed datasets
- Performance is measured using the Concordance Index (C-index), with bootstrap resampling (50 times) to compute 95% confidence intervals for both training and internal validation results
- Risk predictions and SHAP values are averaged across imputations, folds, and repetitions for all individual patients in the internal validation dataset (full-case patients)

<code>c19_survival_explainability.py</code> <br>
  - Beeswarm plot for the most important features for one-year mortality prediction <mark>(averaged SHAP values as input instead of an Explanation object)</mark>
  - Scatter plots with a LOESS curve and a 95% confidence interval  <mark>(averaged SHAP values as input instead of an Explanation object))</mark>
  - Clinically important reference values are highlighted on the LOESS curve when applicable
  - SHAP hazard ratios with 95% confidence interval showing the feature importance for a range of feature values

<code>c19_survival_blobbogram.R</code> <br>
- SHAP values, CIs, and p-values for top ranked clinical predictors, as well as hypertensive medication ACEi and ARB, are visualised in a forest plot
<br>
The code for preprocessing the data is not available due to the sensitivity of the data.
<br>
The overall code documentation was generated using an LLM.
---


## Contact
<address>
<a href="mailto:malin.hjartstrom@med.lu.se">Malin Hjärtström</a><br>
</address> 

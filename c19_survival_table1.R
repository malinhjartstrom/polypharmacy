# =====================================================================
# Script Documentation
# =====================================================================
# Title: Patient Characteristics Summary Table (Table 1)
# Author: malinhjartstrom
# Date: 2025-10-19
# Project: COVID-19 ICU Survival Analysis
#
# Description:
# This script generates a descriptive summary table (Table 1) of baseline
# characteristics for ICU COVID-19 patients. The table compares demographic,
# clinical, and laboratory variables between one-year survivors and
# non-survivors, providing an overview of cohort differences.
#
# =====================================================================
# Requirements
# =====================================================================
# - R version: ≥ 4.2.0
# - Operating System: Cross-platform (Windows/macOS/Linux)
#
# Required Packages:
#   * readxl      (≥ 1.4.3)   – Import Excel files
#   * dplyr       (≥ 1.1.0)   – Data manipulation and transformation
#   * gtsummary   (≥ 1.7.0)   – Create publication-quality summary tables
#   * huxtable    (≥ 5.5.0)   – Convert and export tables (e.g., LaTeX)
#   * purrr       (≥ 1.0.2)   – Functional programming utilities
#   * gt          (≥ 0.10.0)  – Table rendering and export to LaTeX
#
# To install all dependencies:
# install.packages(c("readxl", "dplyr", "gtsummary", "huxtable", "purrr", "gt"))
#
# =====================================================================
# Workflow Summary
# =====================================================================
# 1. **Library Imports** – Load necessary R packages for data import,
#    cleaning, table generation, and export.
#
# 2. **Data Loading** – Import Excel dataset (`c19_survival_data`) and
#    remove redundant index columns.
#
# 3. **Factor Re-Leveling** – Convert selected variables into labeled
#    categorical factors for meaningful presentation.
#
# 4. **Table Creation** – Generate a comparative summary table by one-year
#    survival outcome using `gtsummary::tbl_summary()`, including descriptive
#    statistics, p-values, and missing data percentages.
#
# 5. **Table Formatting & Export** – Apply custom headers, bold labels,
#    and captions. Export the final table to LaTeX (`table1.tex`)
#    for inclusion in manuscripts.
#
# =====================================================================
# Output
# =====================================================================
# - A publication-ready LaTeX file: `table1.tex`
#   (containing summary statistics and comparisons between survivors and
#   non-survivors at one year after ICU admission)
# =====================================================================


# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
library(readxl)     # Read Excel data files
library(gtsummary)  # Generate descriptive summary tables
library(dplyr)      # Data wrangling and transformation
library(huxtable)   # Export tables to LaTeX or Word
library(purrr)      # Functional mapping operations
library(gt)         # Table rendering and export support


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
original <- read_excel("c19_survival_data")          # Load dataset
original <- subset(original, select = -c(...1))      # Remove index column
head(original)                                       # Preview dataset


# ---------------------------------------------------------------------
# Re-level categorical factors for clarity
# ---------------------------------------------------------------------
original$'Dead within one year' <- factor(original$'Dead within one year')
levels(original$'Dead within one year') <- c('Survivors', 'Non-survivors')

original$Sex <- factor(original$Sex)
levels(original$Sex) <- c("Female", "Male")

original$Noradrenaline <- factor(original$Noradrenaline)
levels(original$Noradrenaline) <- c('0', '<= 0.1', '> 0.1')

original$Smoker <- factor(original$Smoker)
levels(original$Smoker) <- c('No, never', 'Yes, previously', 'Yes, presently')


# ---------------------------------------------------------------------
# Generate Table 1 – Patient Characteristics
# ---------------------------------------------------------------------
table1 <- original %>%
  tbl_summary(
    by = 'Dead within one year',
    include = -c('Study month'),
    missing_text = "Missing"
  ) %>%
  add_p() %>%
  modify_table_body(
    ~ .x %>%
      mutate(
        # Add column for percentage of missing values
        missing_values = map_chr(variable, ~ {
          pct <- mean(is.na(original[[.x]])) * 100
          sprintf("%.1f%%", pct)
        })
      )
  ) %>%
  modify_header(
    missing_values ~ "**Missing values**"
  ) %>%
  modify_spanning_header(
    c("stat_1", "stat_2") ~ "**One-year mortality from ICU admission**"
  ) %>%
  modify_caption("**Table 1. Patient Characteristics**") %>%
  bold_labels()


# ---------------------------------------------------------------------
# Export Table to LaTeX (Huxtable and GT)
# ---------------------------------------------------------------------
# Export using huxtable
as_hux_table(table1, include = all_of(everything())) %>%
  huxtable::to_latex()

# Export using gt
gt::gtsave(as_gt(table1), file = "table1.tex")

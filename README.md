# Dengue Risk Prediction and Management Using Explainable Machine Learning

> Project documentation adapted from **Machine Learning Assignment - reviewed.docx**. This README preserves the report’s descriptions and distinguishes results that still require verification.

## Overview

This project forecasts next-week district-level dengue case counts in Sri Lanka using an **Explainable Boosting Machine (EBM)** and presents historical trends, forecasts, and risk status through a web interface.

## Contents

- [Problem definition and dataset](#1-problem-definition-and-dataset-collection)
- [Model selection](#2-selection-of-a-new-machine-learning-algorithm-explainable-boosting-machine-ebm)
- [Training and evaluation](#3-model-training-and-evaluation)
- [Explainability](#4-explainability-and-interpretation)
- [Critical discussion](#5-critical-discussion)
- [Front-end integration](#6-front-end-integration)
- [Running the analysis](#running-the-analysis)
- [Verification notes](#verification-notes)

---


## 1. Problem Definition and Dataset Collection


### 1.1 Problem Description and Relevance

Dengue fever is a serious mosquito-borne viral infection that creates major public health challenges in tropical countries like Sri Lanka. Outbreaks occur frequently and follow seasonal patterns driven by monsoon rains and weather changes.

Predicting future dengue cases helps public health authorities allocate medical supplies, deploy staff to high-risk areas, and execute preventive controls before outbreaks peak. This predictive model forecasts the number of dengue cases for the next week in each district.


### 1.2 Data Source

Original Source: Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka.

Access Link: https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report


### 1.3 Features and Target Variable

Identifiers: year, week, start.date, end.date, and district.

Target Variable: target (the number of dengue cases for the following week).

Engineered Features:

district_code: Numerical code mapped to each district category.

week_sin and week_cos: Sine and cosine transformations of the seasonal week phase to capture yearly monsoon and weather cycles.

cases_lag_1 to cases_lag_4: Past case values from the previous four consecutive weeks.

cases_rolling_avg_4: Moving average of case counts over the last four weeks.


### 1.4 Size of the Dataset

The dataset consists of multi-year weekly records spanning from 2006 onward across multiple districts in Sri Lanka. It contains thousands of rows structured as district-week time-series observations.


### 1.5 Preprocessing Steps

Missing Data Imputation

Checked for missing weeks per district and applied linear interpolation, followed by backward and forward filling to ensure a continuous time series.

Data Cleaning

Standardized column names and sorted records by district and date to preserve chronological order.

Feature Engineering

Transformed date fields into continuous seasonal sine/cosine variables and created lag and rolling average features.

Data Splitting

Divided the data chronologically by year to prepare training and test sets without data leakage.


### 1.6 Ethical Data Use

The dataset contains only aggregated public health surveillance counts grouped by district and week. It contains no personally identifiable information (PII) or sensitive personal records. Patient privacy is fully protected, and no individual consent is needed for this public health analysis.

1.7 Total Cases Across All Districts Per Week

This graph shows the sum of all dengue cases in all districts for each week. You can use this chart to see major disease outbreaks and long-term trends from the historical data. It helps to identify when patient numbers increase across the country.


### 1.8Seasonal Dengue Cases

This graph shows the average number of cases for each week number across all years. You can use this chart to understand yearly patterns. It clearly shows peak periods, which usually match monsoon weather and rain cycles in Sri Lanka.


## 2. Selection of a New Machine Learning Algorithm: Explainable Boosting Machine (EBM)


### 2.1 Algorithm Choice

For this dengue forecasting task, I select the Explainable Boosting Machine (EBM).. This model is a glass-box machine learning model. It gives high prediction accuracy while remaining completely transparent and understandable.


### 2.2 Justification for Selection

Public health authorities require reliable predictions to manage medical resources before outbreaks start. However, they also need to understand why the model predicts high risk in a specific district.

EBM was selected because,

High Accuracy: It uses modern machine learning techniques like bagging and gradient boosting. Its accuracy matches standard tree boosters.

Full Transparency: Every feature has a clear visual shape function. Users can inspect the exact mathematical contribution of each variable (such as past lag cases or seasonal sine/cosine values) to the final prediction.

No Black-Box Limitations: Unlike traditional black-box models, EBM removes the need for external tools like SHAP to explain predictions.


### 2.3 Comparison with Standard Models

Standard Linear and Logistic Regression: These models assume relationships are strictly linear. Dengue transmission follows complex, non-linear seasonal cycles. EBM captures these non-linear patterns easily.

k-Nearest Neighbors (k-NN): k-NN stores all training data and becomes slow when making predictions on large multi-district time-series datasets. EBM builds fast, compact additive functions.

Standard Decision Trees and Random Forests: Single trees overfit easily, while random forests are hard to interpret visually without complex summary tools.


### 2.4 Comparison with LightGBM and XGBoost

While LightGBM and XGBoost are powerful gradient-boosted tree models, they differ from EBM in key ways:

Model Architecture:

LightGBM and XGBoos build deep trees where multiple features interact simultaneously at different splits. This makes it difficult to isolate the exact impact of a single feature.

EBM is a Generalized Additive Model (GAM). It builds individual small trees for each feature separately (and optionally small pairs for interactions).

Interpretability:

LightGBM and XGBoost are black-box models. To explain their predictions, you must use post-hoc approximation tools like SHAP or LIME.

EBM is a glass-box model. Its internal structure is naturally additive, allowing direct inspection of each feature's effect on dengue cases without extra tools.

Performance and Speed:

LightGBM is optimized for raw speed on massive datasets.

EBM trains slightly slower because it cycles through each feature individually during boosting, but it provides equal or better interpretability with competitive accuracy for tabular public health data.


## 3. Model Training and Evaluation


### 3.1 Train/Validation/Test Split Strategy

To evaluate the model properly and prevent data leakage, a chronological time-series split was used:

Training Data: Historical records from prior years (up to the maximum year minus one) were used to train the Explainable Boosting Machine (EBM) model parameters.

Testing Data: The most recent full year of data (max_year) was held out entirely as the test set to simulate real-world future forecasting.


### 3.2 Hyperparameter Choices

The Explainable Boosting Machine (EBM) regressor was configured with optimized hyperparameters tailored for count-based epidemiological data:

objective = "poisson_deviance": Unlike standard squared error, Poisson deviance is specifically designed for count data like disease case counts, handling skewed distributions and variance scaling with the mean much more effectively.

interactions = 10: Allows the model to capture up to 10 pairwise feature interactions (such as the interaction between seasonal timing and past lag cases), improving predictive power while maintaining transparency.

max_interaction_bins = 16: Controls the complexity of interaction pairs to prevent overfitting on sparse bins.

outer_bags = 8: Employs bagging to smooth predictions and reduce variance across models.

learning_rate = 0.015: Uses a slower, more conservative learning rate for finer convergence and robust generalization.

max_rounds = 10000 with early_stopping_rounds = 100: Automatically halts training when validation loss plateaus, protecting against overfitting.

min_samples_leaf = 10: Ensures robust sample sizes in tree leaves for stable splits.


### 3.3 Performance Metrics Used

Because dengue forecasting serves two operational goals (estimating patient volume and triggering outbreak alerts), both regression and classification metrics were used:

Regression Metrics: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) measure the magnitude of prediction errors in actual case counts.  (Coefficient of Determination) evaluates how well variance in future cases is explained by the model.

Classification Metrics: By converting case counts into a binary outbreak risk status using a district-specific 75th percentile threshold, we evaluated Accuracy, Precision, Recall, F1 Score, and ROC-AUC to measure outbreak warning reliability.


### 3.4 Results and Performance Interpretation

3.4.1 Regression Performance

MAE: Pending — populate from notebook/outputs/evaluation_metrics.json after running the updated notebook.

RMSE: Pending — populate from notebook/outputs/evaluation_metrics.json after running the updated notebook.

R²: Pending — populate from notebook/outputs/evaluation_metrics.json after running the updated notebook.

Interpretation: Regression metrics must be checked against a fresh run before making claims about explained variance or error magnitude.

3.4.2 Risk Classification Performance

Accuracy: 0.7368 (73.68%) — derived from the reported confusion matrix; verify against notebook output.

Precision: 0.8095 (80.95%) — derived from TP/(TP+FP).

Recall: 0.7846 (78.46%) — derived from TP/(TP+FN).

F1 Score: 0.7969 (79.69%) — derived from precision and recall.

ROC-AUC: Pending — requires actual continuous prediction scores; cannot be derived from one confusion matrix.

3.4.3 Confusion Matrix Breakdown

True Positives (255): Correctly identified high-risk outbreak weeks.

True Negatives (109): Correctly identified safe baseline weeks.

False Positives (60): Weeks incorrectly flagged as high risk (false alarms).

False Negatives (70): Unpredicted outbreak weeks (missed alerts).

Interpretation: From the reported confusion matrix, precision is 80.95%, recall is 78.46%, and F1 is 79.69%. The 70 false negatives represent missed high-risk weeks. Verify these results against the notebook.

3.4.4 Analysis of the Actual vs. Predicted Graph

Trend Alignment: The line plot comparing actual cases against predicted cases across test records demonstrates that the EBM model accurately traces both baseline lows and sharp epidemic spikes.

Lag Responsiveness: Because the model utilizes 4-week lag features and rolling averages alongside seasonal sine/cosine transformations, it reacts swiftly to changing infection trajectories, successfully anticipating upward case movements prior to outbreak peaks


## 4. Explainability and Interpretation

To understand how the model makes predictions, we use two explainability methods built into the Explainable Boosting Machine (EBM):

Global Feature Importance Analysis: Measures the total effect of each feature on the model predictions across the entire dataset.

Partial Dependence Plots (PDP) / Shape Functions: Shows how changes in a specific feature increase or decrease the predicted number of dengue cases.


### 4.1 Most Influential Features (Global Importance)

Use the updated notebook's feature_importance.png and evaluation_metrics.json to identify and rank the actual model terms. Avoid assuming a ranking before inspecting the generated results.

Recent Lag Cases (cases_lag_1, cases_lag_2): Case counts from the immediately preceding weeks are the most influential predictors. High case numbers in past weeks strongly indicate high case numbers for the next week.

Rolling Average (cases_rolling_avg_4): The four-week moving average helps smooth short-term weekly noise and establishes the general trend direction.

Seasonal Features (week_sin, week_cos): Capture annual weather patterns and monsoon cycles.

District Identity (district_code): Sets the baseline risk level for each region based on population density and local climate conditions.


### 4.2 What the Model Has Learned

The updated notebook plots EBM shape functions for recent case counts, rolling averages, and seasonal terms. Interpret observed patterns after execution. These show associations in the model, not causal effects.

Using Partial Dependence Plots (PDP), we can examine the exact relationship the model learned for each feature:

Infection Momentum: As past lag cases increase, the predicted cases rise in a direct, non-linear pattern, reflecting how outbreaks grow over time.

Seasonal Peaks: The sine and cosine transformations show clear seasonal peaks that correspond directly to historical monsoon rains and high-transmission periods in Sri Lanka.

Regional Baselines: The district feature assigns unique risk intercepts to different locations, accounting for varying population sizes and urban environments.


### 4.3 Alignment with Domain Knowledge

The model's behavior matches known epidemiological facts about dengue transmission:

Epidemic Inertia: Disease outbreaks do not change randomly from week to week. The high weight given to lag features correctly reflects real-world disease momentum.

Weather Dependence: Dengue vector breeding depends on rainfall and humidity. The model successfully learned these yearly cycles through the seasonal features without needing raw weather data.

Geographic Differences: Densely populated and urban districts receive higher baseline risk values, which matches actual public health surveillance records in Sri Lanka.


## 5. Critical Discussion


### 5.1 Limitations of the Model

Lag Dependency: The model relies heavily on past case numbers (cases_lag_1 to cases_lag_4). If a sudden outbreak starts without prior warning signs, the model will miss the initial spike.

Lack of Real-Time Weather Data: The model uses seasonal sine and cosine features instead of real-time weather data. Unusual rain patterns or extreme weather changes are not captured directly.


### 5.2 Data Quality Issues

Interpolation Risks: Missing weeks required linear interpolation and filling. This process can smooth out sharp real-world case changes and hide rapid outbreak starts.

Reporting Delays: Weekly public health surveillance reports can suffer from reporting delays or under-reporting by hospitals, introducing noise into the training data.


### 5.3 Risks of Bias or Unfairness

District-Level Aggregation: Data is grouped by entire districts. This hides micro-level hot spots and neighborhoods with high risk within large districts like Colombo.

Resource Misallocation: If decisions depend only on district totals, large urban zones might receive too many resources while smaller rural areas with severe localized outbreaks receive too few.


### 5.4 Potential Real-World Impact and Ethical Considerations

Proactive Public Health Actions: Accurate predictions help health authorities distribute medical supplies, spray vector breeding sites, and deploy staff before outbreaks peak.

Operational Risks: False alarms waste public funds and create unnecessary worry. Missed outbreaks delay urgent medical help and increase patient risks.

Privacy Protection: Because the dataset contains only aggregated counts per district, no individual patient data is exposed. Patient privacy is fully protected.


## 6 Front-End Integration

The web application provides two main views for users. The dashboard view shows historical case trends and seasonal charts to help track national disease patterns. The prediction view lets users choose a district and a date to calculate predicted case numbers and check risk alert levels for the next week.


### 6.1 Prediction View


### 6.2 Dashboard View

Reproducibility note: Run notebook/dengue_ebm_explained.ipynb from the notebook directory, inspect notebook/outputs/evaluation_metrics.json and PNG figures, and replace the remaining pending values. The original preprocessing can leak future observations through interpolation.


## Running the analysis

The reviewed report references `notebook/dengue_ebm_explained.ipynb`. Run the notebook from the project’s `notebook/` directory with the required Python dependencies installed. It loads project modules and writes figures and evaluation outputs to `notebook/outputs/`.

After execution, inspect `notebook/outputs/evaluation_metrics.json`, `feature_importance.png`, the shape-function plots, the confusion matrix, and the ROC curve. Copy **verified** metrics into the report.

## Verification notes

- **Regression metrics and ROC-AUC are pending** in the reviewed report; do not present them as measured values until the notebook is executed.
- **Confusion-matrix metrics are report-derived**, not independently re-evaluated: TP = 255, TN = 109, FP = 60, FN = 70; accuracy ≈ 73.68%, precision ≈ 80.95%, recall ≈ 78.46%, F1 ≈ 79.69%.
- **Potential look-ahead leakage:** the described linear interpolation and backward fill may use future observations. A chronological train/test split does not eliminate this preprocessing risk.
- **Lag definition:** in the supplied implementation, `cases_lag_1` represents the selected/current week for predicting the following week, not the prior week as the report’s general description suggests.
- **Interpretability caution:** EBM term contributions describe the model’s learned associations; they are not proof of causality. Feature-importance rankings and seasonal peaks should be confirmed from generated plots.

## Source

This README was generated from `Machine Learning Assignment - reviewed.docx`. It is project documentation, not a substitute for independently verified experimental results.

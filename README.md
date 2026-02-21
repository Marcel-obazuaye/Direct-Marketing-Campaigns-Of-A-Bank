# Direct-Marketing-Campaigns-Of-A-Bank
## **Term Deposit Subscription Prediction (R)**
**Overview**
Predictive modeling project to identify bank clients likely to subscribe to a term deposit using demographic, behavioral, and campaign data.
The project focuses on data preprocessing, model comparison, and business insight.

## Objective 
Build and evaluate classification models to predict customer subscription (yes / no) and identify key drivers of conversion.

## **Dataset**
4,521 observations, 16 features
Mixed numeric & categorical variables
Strong class imbalance (~12% subscribers)

## **Methods**
EDA: Missingness analysis, correlation (numeric & categorical)
Preprocessing:
Duplicate removal
MICE imputation (categorical variables)
Feature engineering (pdays transformation)
Scaling & 80/20 train–test split

## **Models:**
Logistic Regression
Random Forest (Conditional Inference Forest)

## **Results**
Model	Accuracy	          Balanced               Accuracy
Logistic Regression	      ~90.4%	               65.7%
Random Forest	            ~89.5%	               64.0%

Logistic Regression slightly outperformed overall
Random Forest provided strong feature importance insights
Both models affected by class imbalance

## **Key Insights**
Call duration, previous campaign outcome, and recency of contact are the strongest predictors
Class imbalance limits subscriber detection → resampling or threshold tuning recommended

## **Tech Stack**
R | dplyr, ggplot2, mice, caret, party, naniar

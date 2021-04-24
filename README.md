# Bank Customer Churn Prediction

## Problem Statement

A Bank wants to take care of customer retention for its product: savings accounts. The bank wants to identify customers likely to churn balances below the minimum balance. The dataset has the customer information such as age, gender, demographics along with their transactions with the bank.
The task would be to predict the propensity to churn for each customer.

## Data Dictionary

There are multiple variables in the dataset which can be cleanly divided into 3 categories:

### I. Demographic information about customers

•	customer_id - Customer id 

•	vintage - Vintage of the customer with the bank in a number of days 

•	age - Age of customer 

•	gender - Gender of customer 

•	dependents - Number of dependents 

•	occupation - Occupation of the customer 

•	city - City of the customer (anonymized) 

### II. Customer Bank Relationship

•	customer_nw_category - Net worth of customer (3: Low 2: Medium 1: High) 

•	branch_code - Branch Code for a customer account 

•	days_since_last_transaction - No of Days Since Last Credit in Last 1 year 

### III. Transactional Information

•	current_balance - Balance as of today 

•	previous_month_end_balance - End of Month Balance of previous month 

•	average_monthly_balance_prevQ - Average monthly balances (AMB) in Previous Quarter 

•	average_monthly_balance_prevQ2 - Average monthly balances (AMB) in previous to the previous quarter 

•	current_month_credit - Total Credit Amount current month 

•	previous_month_credit - Total Credit Amount previous month 

•	current_month_debit - Total Debit Amount current month 

•	previous_month_debit - Total Debit Amount previous month 

•	current_month_balance - Average Balance of current month 

•	previous_month_balance - Average Balance of previous month 

•	churn - Average balance of customer falls below minimum balance in the next quarter (1/0)

## Code and Resources Used

Python version - 3.8.5 (in runtime.txt)
Packages: pandas, numpy, scikit-learn, matplotlib, seaborn, plotly, imbalanced-learn, hyperopt,joblib, and streamlit.
For Web Framework Requirements: pip install -r requirements.txt
Cloud Platform for deployment - Heroku.

## Data Cleaning and EDA

After loading the data, I examined about the nature of features and how they are relevant to target variable through univariate(histogram for numerical and pie charts for categorical features) and bivariate(scatterplots for defining numerical-target relationships and pivot tables for defining categorical-target relationships) analysis.

Splitting of data into train and test.

Removal of missing data using median(for numerical) and mode(for categorical) imputation.

Oultier detection using boxplot and removal using z-score.

Saved the results into separate train and test sets for model building.

## Model Buidling

Identified how many features should be relevant for model using cross validation and f1 score.(it was found to be 7)

Identified those important seven features using Recursive Feature Elimination technique with Decision trees.

As it was found to be an imbalanced dataset,therefore using SVMSMOTE oversampling and Random Under Sampler of imbalanced-learn package to remove class imbalance in engulfing the two into an pipeline.

Scaling of sampled data using RobustScaler.

I tried below classifiers as an baseline using f1 score as an metric on sampled dataset-

Logistic Regression - 0.806 

Decision Tree Classifier - 0.831

XGBoost Classifier - 0.856

Gradient Boosting Classifier - 0.831 

Random Forest Classifier -  0.893

Since Random Forest outperformed all approaches which I assumed before as it is good in case of class imbalance therefore tuning it using hyperopt to increase its performance.

Fitting the tuned model on test dataset giving f1 score of 0.89 in classifying negative class and f1 score of 0.60 in classifying postive class and an ROC score of 0.766.

Saving the model using joblib.

## Productionization

The model was converted into an API using streamlit and deployed in Heroku.

The API takes input through an form built using streamlit and then make predictions in form of probabilites of whether a customer will stay in bank or not.

API link - https://bank-customer-churn-app.herokuapp.com/






 

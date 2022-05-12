# Predictive Analytics and Its Applications in the Investigation of Nursing Home Ratings
> Machine learning project from Spring 2022 Directed Independent Study

> #### ***Further details are found within the report.***

### The Problem (high-level)
For nursing home facilities that participate in Medicare and Medicaid programs, their **overall rating** is calculated from three other ratings, which are the ***health inspections rating***, ***quality measures (QM) rating***, and ***staffing rating***. There are numerous measured/gathered facility-related factors that are used to determine each of the three ratings that then comprise the overall rating.

In this project, we were mainly interested in harnessing the power of machine learning (with Python's Scikit-learn) to investigate **feature importances**.

After *excluding* the calculated ratings of those three specific domains from the nursing home data, we wanted to see if a ML model could find patterns in the data and provide us insights on **which features had the most impact** in predicting the overall rating. 


### The Motivation
Family members research nursing home facilities on the [**Nursing Home Care Compare** website](https://www.medicare.gov/care-compare/?providerType=NursingHome&redirect=true) when deciding on a nursing home for their elderly loved ones, and they first look at the overall star ratings (*ranging from 1 star to 5 stars*).

If a subset of collected nursing home data has the most importance as determined by the ML model, nursing homes can then be **proactive** in terms of not possessing those deficiencies for (unannounced) future inspections in order to obtain a higher overall rating next time. 


### Machine learning dataset
11,585 nursing home instances, 56 numeric features, and the target variable (i.e., overall rating).

![Class label distribution](https://user-images.githubusercontent.com/96803412/168177798-5d841bd7-90c3-48e5-b584-4c84716a85ef.png)

### Classification algorithms investigated
- k-Nearest Neighbors
- Multi-class Logistic Regression
- Random Forest
- AdaBoost

##
### Project Highlights
- Multi-class classification problem
- Restructuring and transforming datasets
- Data merging to create the final ML dataset
- Exploratory data analysis (EDA)
- Scikit-learn pipelines
- Feature scaling with **StandardScaler**
- Missing data imputation with **KNNImputer**
- Feature selection with **SelectKBest**
- 10-fold stratified cross-validation for model evaluation
- Feature importances for the ML models

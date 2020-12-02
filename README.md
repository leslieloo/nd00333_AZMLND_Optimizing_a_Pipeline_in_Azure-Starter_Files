# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used contains data from UCI Bank marketing campaign. We seek to predict if client will subscribe to a term deposit with the bank (indicated as 'yes' in column y). 

Some of the columns used in the prediction includes (Full columns description/metadata can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)):

- **marital** : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
- **default**: has credit in default? (categorical: 'no','yes','unknown')
- **housing**: has housing loan? (categorical: 'no','yes','unknown')
- **loan**: has personal loan? (categorical: 'no','yes','unknown')
- **contact**: contact communication type (categorical: 'cellular','telephone')

Data preparation is needed before the data can be used for model training. Here are some of the data transformation apply on the columns:-
- **marital** - encoded to 1 when value is 'married' or 0 otherwise
- **default** - encoded to 1 when value is 'yes' or 0 otherwise
- **month** - encoded to numeric value from abbreviation (eg. jan as 1)
- **contact** - one-hot encoded to create new column for each category

The data preparation process is provided as **clean_data** function in train.py script.

Overall, the best performing model is model trained using Automated ML (AutoML) with VotingEnsemble (combine multiple ML model such as LightGBM, XGBoostClassifier) as the best ML algorithm. The model accuracy is slighty higher *(Accuracy ~ 0.91505)*, compare to model trained using HyperDrive parameter tuning *(Accuracy ~ 0.90966)*.

## Scikit-learn Pipeline
Model trained using HyperDrive tuning use SKLearn estimator to runs training script (train.py) to train classification model (logistic regression). The training script accepts 2 arguments, Regulaization penalty (--C) and max interation (--max-iter). The training script (train.py) contains data preparation function (clean_data). The trained model is saved for each experiment run.

Other important setting in HyperDrive parameter tuning includes:-
- **RandomParameterSampling** is selected as the parameter sampling method to randomly sample over range of values define in hyperparameter search space. This method greatly reduce the amount of time searching for optimal hyperparameter value, compare with other sampling methods.

- For early termination policy, **BanditPolicy** is used to terminates runs where the primary metric (accuracy) is not within the specified slack factor (0.2) compared to the best performing run.

## AutoML
AutoML used different set of ML algorithm (LightGBM, XGBoostClassifier, RandomForest etc) and hyperparameter tuning to perform model training for each run. The hyperparameter is set automatically based on the ML algorithm being used. The final model VotingEnsemble is generated that combines multiple fitted model. AutoML support automated feature engineering (eg. scaling, normalization, feature selection) which remove lot of manual work that is error prone and redundant.

## Pipeline comparison
Based on the primary metric accuracy, AutoML *(Accuracy ~ 0.91505)* perform much better compare to model trained using Hyperdrive hyperparameter tuning *(Accuracy ~ 0.90966)*. This  could due to to AutoML ability to use different type of model algorithm for training and generate the final ensemble model *VotingEnsemble* which combine all this fitted models. Model trained using Hyperdrive only use 1 ML algorithm (scikit-learn LogisticRegression) as define inside train.py script. AutoML automatically generate and optimize hyperparameter for each ML algorithm during training run, while HyperDrive requires range of hyperparameter values to be defined manually before performing hyperparameter sampling.

Cross validation also implemented as part of AutoML to prevent model overfitting.

## Future work
AutoML do support generating stack ensemble model. This model combines the previous fitted models and trains a meta-model based on this individual models. This may further improve the model performance/accuracy.  

On the other hand, model train using HyperDrive can be further improved by using different advanced classification ensemble algorithm (eg. XGBoost or Random Forest). Other suggested optimization include using Bayesian parameter samplng method to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric



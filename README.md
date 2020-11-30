# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about UCI Bank marketing. We seek to predict if client will subscribe to a term deposit with the bank.

The best performing model is model trained using Automated ML with VotingEnsemble as the best ML algorithm. The model accuracy is slighty higher, compare to model trained using HyperDrive parameter tuning.

## Scikit-learn Pipeline
Model trained using HyperDrive tuning use SKLearn estimator wrapper to runs training script (train.py) to train classification model (logistic regression). The training script accepts 2 arguments, Regulaization penalty (--C) and max interation (--max-iter). The training script (train.py) used scikit-learn LogisticRegression to perform classification.

RandomParameterSampling is used to randomly choose hyperparameter value for training, which reduce training time, with performance/accuracy that can match with GridParameterSampling that will perform all combination of hyperparameter setting given

BanditPolicy is used as the early stopping policy to terminate training process when there is no further improvement compare to the previous run.

## AutoML
AutoML used different set of ML algorithm and hyperparameter tuning to perform training. The hyperparameter is set automatically based on the ML algorithm being used to run training.

## Pipeline comparison
AutoML perform much better compare to model trained using Hyperdrive hyperparameter tuning. This could due to to AutoML ability to use different type of model algorithm to train with hyperparameter tuning, while Hyperdrive only use 1 ML algorithm to perform hyperparameter  tuning.

## Future work
AutoML do support classification task with deep learning algorithm. This may further improve the model accuracy. 



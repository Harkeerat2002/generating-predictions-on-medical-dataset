# CS-433 Project 1
> This README is from the final submission of the project. This repo contains my code and implementation of the first project of the course CS-433 Machine Learning from EPFL. The goal is to build a binary classification model that can predict coronary heart diseases. The original repo can be found [here](https://github.com/CS-433/ml-project-1-mindsync)


This repository contains the code for the first project of the course CS-433 Machine Learning from EPFL. The Project involves using a medical dataset from the Behavioral Risk Factor Surveillance System from Centers for Disease Control USA. The goal is to build a binary classification model that can predict coronary heart diseases.

# The Data

The data can be downloaded from the official github repository of the project:

https://github.com/epfml/ML_course/tree/main/projects/project1/data 

# Scripts
* **costs.py**: contains all the cost functions.
* **gradient_funcs.py**: contains the functions for (stochastic) gradient descent.
* **helpers.py**: contains provided helper functions such as loading the csv data and creating the submission file.
* **implementations.py**: contains the functions of part 1 of the project where we had to implement basic machine learning methods such as linear regression and logistic regression.
* **lin_model_funcs.py**: contains the functions used when considering a linear model.
* **nn.py**: contains the class neural network, initializes a MLP with desidered layers initialized weights.
* **prepare_data.py**: Run this file to prepare the data for further use like deleting features that are unnecessary to the problem.
* **run.py**: Run this file to get predictions to the test and creates a submission file. 
* **scores.py**: Contains the score functions such as accuracy and F1 score.
* **split_funcs.py**: Contains the functions to divided the training set into train and test sets.
* **val_funcs.py**: Contains the functions used for model validation.
* **validate.py**: Run this file to validate the model.


# Prepare the data
Run the script prepare_data.py to delete some chosen features and create npy files of the training set, the test set, the training labels and the test ids. They are stored in the folder "data/dataset/". Please make sure that the files "x_train.csv", "x_test.csv", "y_train.csv" are in the same folder, i.e. "data/dataset/". This file should be run before run.py

# Hyperparameters for the different models:
* **Logistic Regression**: learning rate: 0.3, epochs: 500.
* **Regularized Logistic Regression**: learning rate: 0.01, lambda: 1.3, epochs: 500.
* **Focal Regression**: learning rate: 0.3, gamma: 3.0, alpha: 0.78, epochs: 500.
* **Neural Network with focal loss**: layers: [256, 128, 32]. learning rate: 0.3, epochs: 35, gamma: 2.0, alpha: 0.78, initialization: xavier

# Run.py
Use run.py to run the current best performing model which is the focal regression. Our very best model was achieved by a neural network but exact hyperparameters or training set was lost so we could not reproduce it. The script initializes the model, trains it on the training set and creates predictions for the test set and stores is it in a csv file.





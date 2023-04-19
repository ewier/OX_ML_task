# OX_ML_task

The solution follows the stages outlined in the task description. 

Structure:
The repository contains a file describing the data used:
  - covtype.info
The solution itself is included in the following files:
  - main.py : script to run the overall solution
  - knn.py : contains an implementation of the k-nearest neighbours algorithm (zad 2)
  - baselines.py : contains two ML models used as a baseline (zad 3)
  - neural_network.py : contains a neural network (tensorfow library) (zad 4)
  - utils.py : contains functions used to prepare data (zad 1)

Running the solution:
The file main.py contains function main that runs the solution. Before running the file, please add the datafile covtype.data to the catalogue (it was too large to be uploaded to the repository). Having done that, running the file will cause the results to be shown on the terminal.

Important note:
Unfortunately, due to limited time, I've been unable to complete all of the tasks. My solution covers tasks 1, 2, 3, and part of task 4.


Sources:

KNN:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn-metrics-mean-squared-error

BASELINES:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score

NEURAL NETWORK:
https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

UTILS:
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:35:11 2017

@author: DAVIDGEH
"""

import pandas as pd
#import sklearn
import pdb
import argparse
from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import math
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Code to test xgboost")
parser.add_argument('csv',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--csv', dest='csv', default=None)

args = parser.parse_args()
csv_path = args.csv
# error message is the user does not enter the location of the csv file
if csv_path == None:
    quit("user must specify the location of the training csv file")

# read in the data
master_df = pd.read_csv(csv_path)

# split the data and the target data
data = master_df[['name', 'year_book', 'dob', 'book_published', 'lives_london', \
              'no_siblings', 'no_parents', 'cross_dresses', 'class', 'sex', \
              'period']]
target = master_df['age']

# split test and training dataframe
x_train_with_names, x_test_with_names, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=10)

# drop column 'name'
x_train = x_train_with_names.drop('name', axis = 1)
x_test = x_test_with_names.drop('name', axis = 1)

# parameters for the model
n_estimators = 100
criterion = str('mse')
max_features = 5#math.ceil(np.sqrt(len(x_train.columns) - 1))
max_depth = None
min_samples_leaf = 5 #50

# create the model using random forest method
regr = RandomForestRegressor(n_estimators = n_estimators, # num of trees
                             criterion = criterion, 
                             max_features = max_features, # num of features competing in each node
                             max_depth = max_depth, # max depth of tree
                             min_samples_leaf = min_samples_leaf, # num of observations in the leaf (last node of the tree)
                             random_state = 10, 
                             n_jobs=-1) # num of processors allowed to use

regr.fit(x_train, y_train)
importances = regr.feature_importances_

# calcuate the R^2 of the model on the training set
r_2_train = regr.score(x_train, y_train)
r_2_test = regr.score(x_test, y_test)
print('R^2 of train: %.3f R^2 of test: %.3f' % (r_2_train, r_2_test))

# find the max age difference, the min age difference and the mean age difference
predicted_age_array = regr.predict(x_test)
true_age = pd.DataFrame(y_test, columns = ['age'])

predicted_age = pd.DataFrame(predicted_age_array, index = true_age.index, columns = ['age'])

difference = abs(predicted_age - true_age)

max_age_diff = np.max(difference)
min_age_diff = np.min(difference)
mean_age_diff = np.mean(difference)

features = x_train.columns
indices = np.argsort(importances)

features = x_train.columns
indices = np.argsort(importances)

plt.figure(figsize=(15, 7))
plt.suptitle('Feature Importances for model which has R^2 of train: %.2f and R^2 of test: %.2f' % (r_2_train, r_2_test))
plt.title('n_estimators = %.0f, criterion = %s, max_features = %.0f max_depth = None, min_samples_leaf = %.0f' % (n_estimators, criterion, max_features, min_samples_leaf))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices]) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

pdb.set_trace()

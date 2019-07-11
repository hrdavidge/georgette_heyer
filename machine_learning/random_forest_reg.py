# looking at random forest regressor using the Heyer character dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Code to test random forest regressor")
parser.add_argument('csv',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--csv', dest='csv', default=None)

args = parser.parse_args()
csv_path = args.csv
# error message is the user does not enter the location of the csv file
if csv_path == None:
    quit("user must specify the location of the training csv file")

# read in the data
master_df = pd.read_csv(csv_path)

# check for NaNs in the data
print 'Checking for NaNs', master_df.isnull().sum()

# split the data and the target data
data = master_df[['name', 'year_book', 'dob', 'book_published', 'lives_london', \
              'no_siblings', 'no_parents', 'cross_dresses', 'class', 'sex', \
              'period']]
target = master_df['age']

# split test and training dataframe
x_train_with_names, x_test_with_names, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)

# drop column 'name'
x_train = x_train_with_names.drop('name', axis = 1)
x_test = x_test_with_names.drop('name', axis = 1)

# parameters for the model
n_estimators = 5#100
criterion = str('mse')
max_features = 3
max_depth = None
min_samples_leaf = 5
oob_score = True

# create the model using random forest method
regr = RandomForestRegressor(n_estimators = n_estimators, # num of trees
                             criterion = criterion,
                             max_features = max_features, # num of features competing in each node
                             max_depth = max_depth, # max depth of tree
                             min_samples_leaf = min_samples_leaf, # num of observations in the leaf (last node of the tree)
                             random_state = 10,
                             #oob_score=oob_score, # if using out of the bag score
                             n_jobs=-1) # num of processors allowed to use

regr.fit(x_train, y_train) # training the model

# looking at the attributes
estimators = regr.estimators_ # description of each tree

importance = regr.feature_importances_ # an array of the fractional importance of the each feature

num_features = regr.n_features_ # the number of features

num_outputs = regr.n_outputs_ # the number of outputs when the model is built

#oob_score = regr.oob_score_ # score the training dataset using an out-of-bag estimator, this computes the average of correct classifications
# basically the coefficent of determination of R**2 using 'unseen' data not used to build the model

#oob_predict = regr.oob_prediction_ # The prediction for the values of training dataset using the oob method

# now having a look at the methods
leaf_indices = regr.apply(x_test) # get the numbers of the all the leaves the test dataset ends up in

decision_path = regr.decision_path(x_test)

parameters = regr.get_params() # the parameters of the model

predicted_age_array = regr.predict(x_test) # running the test dataset through the model, giving an array of predicted values

r_2_train = regr.score(x_train, y_train) # calculating the R squared of the train dataset
r_2_test = regr.score(x_test, y_test) # calculating the R squared of the test dataset

set_params = regr.set_params() # set the parameters for the model

# print the R squared
print('R^2 of train: %.3f R^2 of test: %.3f' % (r_2_train, r_2_test))

# find the max age difference, the min age difference and the mean age difference

true_age = pd.DataFrame(y_test, columns = ['age'])

predicted_age = pd.DataFrame(predicted_age_array, index = true_age.index, columns = ['age'])

difference = abs(predicted_age - true_age)

max_age_diff = np.max(difference)
min_age_diff = np.min(difference)
mean_age_diff = np.mean(difference)

features = x_train.columns
indices = np.argsort(importance)

plt.figure(figsize=(15, 7))
plt.suptitle('Feature Importances for model which has R^2 of train: %.2f and R^2 of test: %.2f' % (r_2_train, r_2_test))
plt.title('n_estimators = %.0f, criterion = %s, max_features = %.0f max_depth = None, min_samples_leaf = %.0f' % (n_estimators, criterion, max_features, min_samples_leaf))
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices]) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

pdb.set_trace()

# looking at linear regression using the Heyer character dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

parser = argparse.ArgumentParser("Code to test linear regression")
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

# build the model
reg = LinearRegression()

reg.fit(x_train, y_train) # training the model

# the attributions of the model
coeff = reg.coef_ # the coefficients of the model
intercept = reg.intercept_ # the intercept of the model

# now having a look at the methods
parameters = reg.get_params() # the parameters of the model

predicted_age_array = reg.predict(x_test) # running the test dataset through the model, giving an array of predicted values

r_2_train = reg.score(x_train, y_train) # calculating the R squared of the train dataset
r_2_test = reg.score(x_test, y_test) # calculating the R squared of the test dataset

set_params = reg.set_params() # set the parameters for the model

# print the R squared
print('R^2 of train: %.3f R^2 of test: %.3f' % (r_2_train, r_2_test))

# find the MSE
mse = mean_squared_error(y_test, predicted_age_array)
# find the MAE
mae = mean_absolute_error(y_test, predicted_age_array)

print('The mean squared error is: %.3f, and the mean absolute error is: %.3f' % (mse, mae))

# find the max age difference, the min age difference and the mean age difference
true_age = pd.DataFrame(y_test, columns = ['age'])

# find the max and min difference
predicted_age = pd.DataFrame(predicted_age_array, index = true_age.index, columns = ['age'])
difference = abs(predicted_age - true_age)
max_age_diff = np.max(difference)
min_age_diff = np.min(difference)

print('The minimum age difference is: %.3f, and the maximum age difference is: %.3f' % (min_age_diff, max_age_diff))

pdb.set_trace()

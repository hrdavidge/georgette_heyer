# looking at SVM for regression using the Heyer character dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


parser = argparse.ArgumentParser("Code to test SVM regression")
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
reg = SVR(kernel='linear')

# train the model
reg.fit(x_train, y_train)

# looking at attributes
support_vectors_ind = reg.support_ # the indicies of the support vectors
support_vectors = reg.support_vectors_ # the support vectors
support_vector_coef = reg.dual_coef_ # coefficients for the support vector in the decision function
feature_weights = reg.coef_ # feature weights, this only works when using a linear kernel
intercept = reg.intercept_ # the intercept corrdinates

# looking at the methods
get_params = reg.get_params()
prediction = reg.predict(x_test)
train_score = reg.score(x_train, y_train)
test_score = reg.score(x_test, y_test)

print('The accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

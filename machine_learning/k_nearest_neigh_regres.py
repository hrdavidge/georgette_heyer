# testing out k nearest neighbours for regression on the Heyer character dataset
# importing the libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


parser = argparse.ArgumentParser("Code to test k nearest neighbours regression")
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

# we need to convert all the 2s to 0 (male)
target = target.replace([2], [0])
# so 0 = male and 1 = female

# split test and training dataframe
x_train_with_names, x_test_with_names, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)

# drop column 'name'
x_train = x_train_with_names.drop('name', axis = 1)
x_test = x_test_with_names.drop('name', axis = 1)

# parameters
n_neighbors = 5
# building the model
neigh = KNeighborsRegressor(n_neighbors=n_neighbors)

# training the model
neigh.fit(x_train, y_train)

# looking at the methods
get_params = neigh.get_params() # returns the parameters of the model
kneighbours = neigh.kneighbors(x_test.head(1), n_neighbors = n_neighbors) # the first array gives the distance between the new data point and the k neighbours, and the second array gives the sample number of the k neighbours
kneighbours_graph = neigh.kneighbors_graph(x_test.head(1), n_neighbors = n_neighbors, mode = 'distance') # returns a sparce matrix for the k neighbours for the new data points
prediction_array = neigh.predict(x_test) # predicted test values
train_score = neigh.score(x_train, y_train) # the mean auuracy of the training dataset
test_score = neigh.score(x_test, y_test) # the mean acccuracy for the test dataset

print('The mean accuracy of the train dataset is: %.3f and the mean accuracy of the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

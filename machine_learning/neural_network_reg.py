# testing out neural networks for regression using the Georgette Heyer characters dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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

# building the model
# setting the model parameters
hidden_layer_sizes = (100,)
solver = 'lbfgs'
random_state = 42

mlpr = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes, solver = solver, random_state = random_state)

# training the model
mlpr.fit(x_train, y_train)

# looking at the attributes
loss = mlpr.loss_ # the loss computed with the loss function
coefs = mlpr.coefs_ # a list of the weight matrix for each layer
intercepts = mlpr.intercepts_ # a list of the bias vector for each layer
n_iter = mlpr.n_iter_ # the number of interations the solver has run
n_layers = mlpr.n_layers_ # the number of layers of the model
n_outputs = mlpr.n_outputs_ # the number of outputs
out_activation = mlpr.out_activation_ # the name of the output activation function used

# looking at the methods
get_params = mlpr.get_params() # returning the parameters for the model
predition_array = mlpr.predict(x_test) # running the test dataset through the model, giving an array of predicted values
train_score = mlpr.score(x_train, y_train) # returns the mean accuracy of the training set
test_score = mlpr.score(x_test, y_test) # returns the mean accuracy of the test set

print('The R^2 score for the train dataset is: %.3f and the R^2 for the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

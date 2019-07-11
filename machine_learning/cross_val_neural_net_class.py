# testing cross validation and grid search using neural network for classification using the Georgette Heyer survey dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

parser = argparse.ArgumentParser("Code to test cross validation, grid search and neural network classifcation model")
parser.add_argument('csv',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--csv', dest='csv', default=None)

args = parser.parse_args()
csv_path = args.csv
# error message is the user does not enter the location of the csv file
if csv_path == None:
    quit("user must specify the location of the training csv file")

# read in the data
master_df = pd.read_csv(csv_path)

# just get the columns we want
df = master_df[['fav_novel1', 'adventure', 'bio', 'childrens', 'classic', 'comic', 'comics',\
                 'crime', 'dystopian', 'erotica', 'fan', 'fairy', 'fantasy', 'folklore', \
                  'gothic', 'historical_f', 'historical_r', 'steampunk', 'horror', \
                  'epic', 'modern', 'romance2', 'science', 'short', 'suspense', 'teen']]

# let's drop all rows with NaN in fav_novel1
# but the fact the user did not fill out the other columns may be significate, so set the other NaNs to -1
df = df[pd.notnull(df['fav_novel1'])]
# setting NaN values in genre types to -1
df = df.fillna(value=-1)

# split the data and the target data
target = df['fav_novel1']
data = df.drop(['fav_novel1'], axis=1)

# now we need to convert all the columns in data to categorical

def convert_to_category(column):
    return column.astype('category')

# data columns names
data_col_names = data.columns.values

# loop through each columns
for name in data_col_names:
    data[name] = data[name].astype('category')

# split test and training dataframe
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# building the model
# setting the model parameters

hidden_layer_sizes = (500,)
solver = 'lbfgs'
random_state = 42

mlpc = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, solver = solver, random_state = random_state)

# train the model
mlpc.fit(x_train, y_train)

# looking at the attributes
classes = mlpc.classes_ # target column classes
loss = mlpc.loss_ # the loss computed with the loss function
coefs = mlpc.coefs_ # a list of the weight matrix for each layer
intercepts = mlpc.intercepts_ # a list of the bias vector for each layer
n_iter = mlpc.n_iter_ # the number of interations the solver has run
n_layers = mlpc.n_layers_ # the number of layers of the model
n_outputs = mlpc.n_outputs_ # the number of outputs, maybe corresponding to the number of classes
out_activation = mlpc.out_activation_ # the name of the output activation function used

# looking at the methods
get_params = mlpc.get_params() # returning the parameters for the model
prediction_array = mlpc.predict(x_test) # running the test dataset through the model, giving an array of predicted values
predict_log_proba = mlpc.predict_log_proba(x_test) # log of probability estimate for each class
predict_proba = mlpc.predict_proba(x_test) # the probability for each class
train_score = mlpc.score(x_train, y_train) # returns the mean accuracy of the training set
test_score = mlpc.score(x_test, y_test) # returns the mean accuracy of the test set

print('Using the standard neural network model the accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

# performing cross validation
cross_val_score = cross_val_score(mlpc, x_train, y_train, cv = 3)
# this returns an array of estimators for each iteration using .score()

# setting the parameters
param_grid = {'hidden_layer_sizes': [(100,), (200, ), (300, ), (400, ), (500,)]}
cv = 3

gs_mlpc = GridSearchCV(mlpc, param_grid = param_grid, cv = cv)

# fitting the model
gs_mlpc.fit(x_train, y_train)

# looking at the attributes
cv_results = gs_mlpc.cv_results_ # information for a dataframe giving in formation about the grid search
cv_results_df = pd.DataFrame(cv_results) # a dataframe giving in formation about the grid search
best_estimator = gs_mlpc.best_estimator_ # giving the parameters of the model which give the
best_score = gs_mlpc.best_score_ # the Mean cross-validated score of the best_estimator
best_params = gs_mlpc.best_params_ # the parameter settings which gave the best score
best_index = gs_mlpc.best_index_ # the index of the cv_results arrays which gives the best score
scorer = gs_mlpc.scorer_ # the scorer funciton used on the hold out data
num_splits = gs_mlpc.n_splits_ # the number of cross validation splits
refit_time = gs_mlpc.refit # number of seconds used to fit best model to the dataframe

# looking at the methods
#decision_function = gs_mlpc.decision_function(x_train)
get_params = gs_mlpc.get_params() # returns the parameters of the model
#inverse_transform = gs_mlpc.inverse_transform(x_train)
prediction_array = gs_mlpc.predict(x_test) # finding the predicted values for the model with the best parameters
prediction_log_proba = gs_mlpc.predict_log_proba(x_test) # returns the predicted log probability for the model with the best parameters
prediction_proba = gs_mlpc.predict_proba(x_test) # returns the predicted probabiloty for the model with the best parameters
train_score = gs_mlpc.score(x_train, y_train) # returns the mean accuracy of the training set
test_score = gs_mlpc.score(x_test, y_test) # returns the mean accuracy of the test set
#transformed_estimator = gs_mlpc.transform(x_test)

print('Using the tuned neural network model the accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

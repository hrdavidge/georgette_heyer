# testing neural network for classification using the Georgette Heyer survey dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser("Code to test neural network classifcation model")
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

print('The accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

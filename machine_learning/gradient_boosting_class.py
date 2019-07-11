# testing gradient boosting classification using the Georgette Heyer survey dataset
# import the libraries
from sklearn.ensemble import GradientBoostingClassifier
import pdb
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("Code to test gradient boosting classifer model")
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
# data columns names
data_col_names = data.columns.values

# loop through each columns
for name in data_col_names:
    data[name] = data[name].astype('category')

# split test and training dataframe
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# parameters for the model
loss = 'deviance'
learning_rate = 0.1
n_estimators = 100
criterion = str('friedman_mse')
max_features = 5
max_depth = 3
min_samples_leaf = 5

# create the model using random forest method
clf = GradientBoostingClassifier(loss = loss, # the differentiable loss function used
                             learning_rate = learning_rate, # the scaled contribution of each tree
                             n_estimators = n_estimators, # num of trees
                             criterion = criterion,
                             max_features = max_features, # num of features competing in each node
                             max_depth = max_depth, # max depth of tree
                             min_samples_leaf = min_samples_leaf, # num of observations in the leaf (last node of the tree)
                             random_state = 42)

clf.fit(x_train, y_train) # building the gradient boosting random forest using the training dataset

predicted_array = clf.predict(x_test) # running the test set through the model

# calcuate the R^2 of the model on the training set
r_2_train = clf.score(x_train, y_train)
r_2_test = clf.score(x_test, y_test)
print('R^2 of train: %.3f R^2 of test: %.3f' % (r_2_train, r_2_test))

importance = clf.feature_importances_ # an array containing the fractional importance of each feature
data = {'true_value': y_test, 'predicted_value': predicted_array}

predicted_true_testset = pd.DataFrame(data = data, columns = ['true_value', 'predicted_value'])

features = x_train.columns
indices = np.argsort(importance)

plt.figure(figsize=(15, 7))
plt.suptitle('Feature Importances for model which has R^2 of train: %.2f and R^2 of test: %.2f' % (r_2_train, r_2_test))
plt.title('n_estimators = %.0f, criterion = %s, max_features = %.0f max_depth = None, min_samples_leaf = %.0f' % (n_estimators, criterion, max_features, min_samples_leaf))
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()

pdb.set_trace()





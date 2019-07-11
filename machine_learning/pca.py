# code to test pca
# first perform a standard random forest
# then do dimensionality reduction performing pca
# then re-run the random forest model
# note a random forest model is not the best model to use pca on, as a random forest requires a large number of variables
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # for use to normalise the variables

parser = argparse.ArgumentParser("Code to study pca using a random forest classifcation model")
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

# the variables are actually categorical, but as they are nan, no, yes
# for the purpose of this exercise we can treat them as numerical
# running the model with them as categorical or numerical gives about the same R^2
# now we need to convert all the columns in data to categorical
#def convert_to_category(column):
#    return column.astype('category')

# data columns names
data_col_names = data.columns.values

# loop through each columns
#for name in data_col_names:
#    data[name] = data[name].astype('category')

# split test and training dataframe
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# parameters for the model
n_estimators = 100
criterion = str('gini')
max_features = 5
max_depth = None
min_samples_leaf = 5
oob_score = True

# create the model using random forest method
clf = RandomForestClassifier(n_estimators = n_estimators, # num of trees
                             criterion = criterion,
                             max_features = max_features, # num of features competing in each node
                             max_depth = max_depth, # max depth of tree
                             min_samples_leaf = min_samples_leaf, # num of observations in the leaf (last node of the tree)
                             random_state = 10,
                             n_jobs=-1, # num of processors allowed to use
                             oob_score = oob_score) # computes the average of correct classifications

clf.fit(x_train, y_train) # building the random forest using the training dataset

predicted_array = clf.predict(x_test) # running the test set through the model

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

print('Using Random Forest, the accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

# next we want to perform pca to reduce the dimensions of the data

# remember the variables need to be normalised, as they only take the values -1, 0 or 1 I will not need to do so in this case

num_variables = x_train.shape
n_components = num_variables[1]
random_state = 42

pca = PCA(n_components = n_components, random_state = random_state)

# reduce the dimensions of the training dataset
pca.fit(x_train)

# looking at the attributes
components = pca.components_ # the eigenvectors sorted by associated eigenvalue value in descending variance order. Also the principal components
explained_variance = pca.explained_variance_ # eigenvalues in descending order of variance. Also the variance os each principal component
explained_variance_ratio = pca.explained_variance_ratio_ # eigenvalues as a fraction of the sum of eigenvalues in descending order of variance
singular_values = pca.singular_values_ # equal to the 2-norms of the eigenvectors in the lower dimensional space
mu = pca.mean_ # the feature mean estimated from the training set
num_components = pca.n_components_ # an estimated number of components
noise_variance = pca.noise_variance_ # The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop

# looking at the methods
feature_transform = pca.fit_transform(x_train, y_train) # the transformed dataset
data_covariance = pca.get_covariance() # the covariance matrix
get_params = pca.get_params() # the parameters of the pca model
precision_matrix = pca.get_precision() # the precision matrix - the inverse of the covariance matrix
inverse_transform = pca.inverse_transform(feature_transform) # the inverse transformation of the variables - back to their original space
train_score = pca.score(x_train, y_train)
train_log_score = pca.score_samples(x_train)
test_transform = pca.transform(x_test) # apply the dimensionality reduction

print'The fractional contribution to the sum of the eigenvalues:'
print(explained_variance_ratio)
print'We are just going to keep the principal components which contribution at least 5%'
print'Thus we are going to have 6 principal components'


# re-running the pca with just 6 components
n_components = 6 #num_variables[1]
random_state = 42

pca = PCA(n_components = n_components, random_state = random_state)

# reduce the dimensions of the training dataset
pca.fit(x_train)
# now transform the training dataset
feature_transform = pca.fit_transform(x_train, y_train)
# now transform the test dataset
test_transform = pca.transform(x_test)

# now re-build the random forest model using the dimesionally reduced training dataset
# parameters for the model
n_estimators = 100
criterion = str('gini')
max_features = 5
max_depth = None
min_samples_leaf = 5
oob_score = True

clf_pca = RandomForestClassifier(n_estimators = n_estimators,
                             criterion = criterion,
                             max_features = max_features,
                             max_depth = max_depth,
                             min_samples_leaf = min_samples_leaf,
                             random_state = 10,
                             n_jobs=-1,
                             oob_score = oob_score)

# train the new model
clf_pca.fit(feature_transform, y_train)

# run the testset though the new model
predicted_array_pca = clf_pca.predict(test_transform)

# evaluating the model
train_score_pca = clf_pca.score(feature_transform, y_train)
test_score_pca = clf_pca.score(test_transform, y_test)

print('Using Random Forest, the accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score_pca, test_score_pca))

pdb.set_trace()

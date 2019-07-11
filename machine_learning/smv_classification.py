# testing SVM classification using the Georgette Heyer survey dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


parser = argparse.ArgumentParser("Code to test support vector machines for classification")
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

# data columns names
data_col_names = data.columns.values

# loop through each columns
for name in data_col_names:
    data[name] = data[name].astype('category')

# split test and training dataframe
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)

# build the model
clf = SVC(kernel='linear')

# train the model
clf.fit(x_train, y_train)

# look at the attributes
support_vectors_ind = clf.support_ # the indicies of the support vectors
support_vectors = clf.support_vectors_ # the support vectors
num_support_vectors = clf.n_support_ # the number of support vectors for each target class
support_vector_coef = clf.dual_coef_ # coefficients for the support vector in the decision function
feature_weights = clf.coef_ # feature weights
intercept = clf.intercept_ # the intercept corrdinates
correctly_fitted = clf.fit_status_ # if correctly fitted = 0, else 1
prob_a = clf.probA_ # assigned a value of probability = True
prob_b = clf.probB_ # assigned a value of probability = True

# looking at the methods
decision_function = clf.decision_function(x_train) # returns the decision function
get_params = clf.get_params() # return the parameters of the model
prediction = clf.predict(x_test) # testing the model

train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)

print('The accuracy score for the train dataset is: %.3f and the accuracy for the test dataset is: %.3f' % (train_score, test_score))

pdb.set_trace()

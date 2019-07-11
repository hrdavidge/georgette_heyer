# testing discriminate analysis using the Georgette Heyer survey dataset
# importing libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


parser = argparse.ArgumentParser("Code to test discrimate analysis")
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
clf = LinearDiscriminantAnalysis()
# train the model
clf.fit(x_train, y_train)

# looking at the attributes
coef = clf.coef_
intercept = clf.intercept_
#covariance_mat = clf.covariance_ # gives the covariance matrix, does not work for the solver 'svd'
perc_vari = clf.explained_variance_ratio_
means = clf.means_
priors = clf.priors_
scalings = clf.scalings_
overall_mean = clf.xbar_
classes = clf.classes_

# looking at the methods
decison_function = clf. decision_function(x_test)
fit_transform = clf.fit_transform(x_test, y_test)
get_params = clf.get_params()
prediction = clf.predict(x_test)
predict_log_proba = clf.predict_log_proba(x_test)
predict_proba = clf.predict_proba(x_test)
mean_accuracy_train = clf.score(x_train, y_train)
mean_accuracy_test = clf.score(x_test, y_test)
transform = clf.transform(x_test)

print('The mean accuracy of the train dataset is: %.3f and the mean accuracy of the test dataset is: %.3f' % (mean_accuracy_train, mean_accuracy_test))

pdb.set_trace()

# looking at k-means using the Heyer character dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser("Code to test K means")
parser.add_argument('csv',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--csv', dest='csv', default=None)

args = parser.parse_args()
csv_path = args.csv
# error message is the user does not enter the location of the csv file
if csv_path == None:
    quit("user must specify the location of the training csv file")

# read in the data
master_df = pd.read_csv(csv_path)

data = master_df.drop('novel', axis = 1)

# check for NaNs in the data
print 'Checking for NaNs', data.isnull().sum()

# split test and training dataframe
x_train_with_names, x_test_with_names = train_test_split(data, test_size=0.10, random_state=42)

# drop column 'name'
x_train = x_train_with_names.drop('name', axis = 1)
x_test = x_test_with_names.drop('name', axis = 1)

# setting the model parameters
init = 'random'
n_init = 10
max_iter = 100
random_state = 42

# now to perform the elbow test to see how many clusters I need
Ks = range(1, 10)
km = [KMeans(n_clusters = n, init = init, n_init = n_init, max_iter = max_iter, random_state = random_state) for n in Ks]
distorsions = [km[i].fit(x_train).inertia_ for i in range(len(km))]

fig = plt.figure(figsize=(15, 5))
plt.plot(Ks, distorsions)
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia Attribute')
plt.title('Elbow curve')
plt.show()

# two or three clusters look best

# building the model
n_clusters = 3

kmeans = KMeans(n_clusters = n_clusters, init = init, n_init = n_init, max_iter = max_iter, random_state = random_state)

# training the model
kmeans.fit(x_train)

# looking at the attributes
cluster_centers = kmeans.cluster_centers_ # the feature values for the centres of the clusters
labels = kmeans.labels_ # number of cluster each training dataset is assigned to
inertia = kmeans.inertia_ # sum of squared distances between data points and their cluster centre
n_iter = kmeans.n_iter_ # number of iternation runs

# looking at the methods
fit_predict = kmeans.fit_predict(x_train) # looks like labels_ the cluster number for each training dataset
fit_transform = kmeans.fit_transform(x_train) # perform clustering and convert x_train to a cluster-distance space
get_params = kmeans.get_params() # returns the model parameters
prediction = kmeans.predict(x_test) # running the test set through the model
train_score = kmeans.score(x_train) # Opposite of the value of the training dataset on the K-means objective
test_score = kmeans.score(x_test) # Opposite of the value of the test dataset on the K-means objective
transform = kmeans.transform(x_train) # convert x_train to cluster-distance space

# I am interested to see how the clustering worked
# attached labels_ to train dataset
new_train_df = x_train_with_names
new_train_df['labels'] = labels

# now I want to look at each cluster
array_of_clusters = []
for i in np.unique(labels):
    array_of_clusters.append(new_train_df.loc[new_train_df['labels'] == i])

pdb.set_trace()

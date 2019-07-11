# testing random forest classification using the Georgette Heyer survey dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

parser = argparse.ArgumentParser("Code to test random forest classifcation model")
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

# looking at the attributes
estimators = clf.estimators_ # a summary of each of the trees or sub-estimators

classes = clf.classes_ # unique list of target column values

n_classes = clf.n_classes_ # the number of unique values in the target column

n_features = clf.n_features_ # the number of features

n_outputs = clf.n_outputs_ # the number of outputs when the model is built

importance = clf.feature_importances_ # an array containing the fractional importance of each feature

oob_score = clf.oob_score_ # score the training dataset using an out-of-bag estimator, this computes the average of correct classifications
# basically the coefficent of determination of R**2 using 'unseen' data not used to build the model

oob_decision_func = clf.oob_decision_function_

# now looking at the methods
leaf_indicies = clf.apply(x_test) # Using apply - which says which end leaf each row in x_test ends in

# using decision_path -
indicator, n_nodes_ptr = clf.decision_path(x_test)

parameters = clf.get_params()

predicted_array = clf.predict(x_test) # running the test set through the model

log_mean_predicted_class = clf.predict_log_proba(x_test)

mean_predicted_class = clf.predict_proba(x_test)

mean_accuracy = clf.score(x_test, y_test) # returns the accuracy (coefficent of determination of R**2) of the predicted test data outputs and the true values of the test data


# calcuate the R^2 of the model on the training set
r_2_train = clf.score(x_train, y_train)
r_2_test = clf.score(x_test, y_test)
print('R^2 of train: %.3f R^2 of test: %.3f' % (r_2_train, r_2_test))

data = {'true_value': y_test, 'predicted_value': predicted_array}

predicted_true_testset = pd.DataFrame(data = data, columns = ['true_value', 'predicted_value'])

features = x_train.columns
indices = np.argsort(importance)

plt.figure(figsize=(15, 7))
plt.suptitle('Feature Importances for model which has R^2 of train: %.2f and R^2 of test: %.2f' % (r_2_train, r_2_test))
plt.title('n_estimators = %.0f, criterion = %s, max_features = %.0f max_depth = None, min_samples_leaf = %.0f' % (n_estimators, criterion, max_features, min_samples_leaf))
#plt.title('n_estimators = %.0f, criterion = %s, max_features = %.0f max_depth = None, min_samples_leaf = %.0f' % (n_estimators, max_features, min_samples_leaf))
plt.barh(range(len(indices)), importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices]) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()

confuse_mat = confusion_matrix(y_test, predicted_array)


# below is from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize,
                          title,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = unique_labels(classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    print'finished!'
    return ax


np.set_printoptions(precision=2)

class_names = y_test.unique()

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, predicted_array, classes=class_names, normalize=False,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, predicted_array, class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()


pdb.set_trace()
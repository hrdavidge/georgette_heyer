# predicting the reader's favourite Heyer novel given the reading genres they like
# importing libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels

parser = argparse.ArgumentParser("Code to test multinomial logistic regression")
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
clf = LogisticRegression(penalty = 'l2', #
                        random_state = 10,
                        solver = 'lbfgs', #
                         multi_class = 'multinomial') # stating this is a binary problem)

# training the model
clf.fit(x_train, y_train)

# attributes
classes = clf.classes_ # list of class labels
coeff = clf.coef_ # coefficients of the model
intercept = clf.intercept_ # the intercept for each class
n_iter = clf.n_iter_ # the number of iterations for each class - in the binary case it only returns one value

# now having a look at the methods
dec_func = clf.decision_function(x_test) # the confidence score for each test data
density = clf.densify() # returns the coeffient matrix in densy array format
get_param = clf.get_params() # returns the hyper-parameters
predicted_array = clf.predict(x_test) # running the test dataset through the model, giving an array of predicted values
predic_log_proba = clf.predict_log_proba(x_test) # log of probabily estimate for each class
predic_prob = clf.predict_proba(x_test) # the probability for each class
mean_accuracy = clf.score(x_test, y_test) # returns the mean accuracy of the test set
sparsify = clf.sparsify() # returns the coeffient matrix in sparse format

print('The mean accuracy of the test set is: %.3f' % mean_accuracy)

# the confusion matrix
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

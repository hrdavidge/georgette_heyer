# looking at logistic regression using the Heyer character dataset
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels

parser = argparse.ArgumentParser("Code to test binary logistic regression")
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
              'no_siblings', 'no_parents', 'cross_dresses', 'class', 'age', \
              'period']]
target = master_df['sex']

# split test and training dataframe
x_train_with_names, x_test_with_names, y_train, y_test = train_test_split(data, target, test_size=0.10, random_state=42)

# drop column 'name'
x_train = x_train_with_names.drop('name', axis = 1)
x_test = x_test_with_names.drop('name', axis = 1)

# build the model
clf = LogisticRegression(random_state = 10,
                        solver = 'liblinear', # for use with small datasets
                         multi_class = 'ovr') # stating this is a binary problem)

# training the model
clf.fit(x_train, y_train)

# attributes
classes = clf.classes_ # list of class labels
coeff = clf.coef_ # coefficients of the model
intercept = clf.intercept_ # the intercept of the model
n_iter = clf.n_iter_ # the number of iterations for each class - in the binary case it only returns one value

# now having a look at the methods
dec_func = clf.decision_function(x_test) # the confidence score for each test data
density = clf.densify() # returns the coeffient matrix in densy array format
get_param = clf.get_params() # returns the hyper-parameters
predicted_array = clf.predict(x_test) # running the test dataset through the model, giving an array of predicted values
predic_log_proba = clf.predict_log_proba(x_test) # log of probability estimate for each class
predic_prob = clf.predict_proba(x_test) # the probability for each class
mean_accuracy = clf.score(x_test, y_test) # returns the mean accuracy of the test set
sparsify = clf.sparsify() # returns the coeffient matrix in sparse format

print('The mean accuracy of the test set is: %.3f' % mean_accuracy)

# now findng the confusion matrix for the data
# we first need to convert the 1 and 2 to 'female' and 'male'
y_test = y_test.replace([1, 2], ['female', 'male'])
# converting predictions from numpy array to pandas series
predicted_series = pd.Series(data = predicted_array)
predicted_series = predicted_series.replace([1, 2], ['female', 'male'])
# the confusion matrix
confuse_mat = confusion_matrix(y_test, predicted_series)

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
plot_confusion_matrix(y_test, predicted_series, classes=class_names, normalize=False,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, predicted_series, class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
pdb.set_trace()

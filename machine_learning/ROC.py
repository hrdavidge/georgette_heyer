# testing out ROC and AUROC - using the GH character dataset and testing whether they are male of female
# import libraries
import pandas as pd
import pdb
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Code to test ROC curve")
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

# we need to convert all the 2s to 0 (male)
target = target.replace([2], [0])
# so 0 = male and 1 = female

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

# run the test dataset through the model and give the percentage score for each the data being of each class
class_probability = clf.predict_proba(x_test)
# putting the class probabiities into a dataframe
class_proba_df = pd.DataFrame(class_probability, columns=clf.classes_)

# now create an ROC curve of the results
fpr, tpr, thresholds = roc_curve(y_test, class_proba_df[1], pos_label=1)

# finding the AUC area under the curve
# using the predictions
auc_score = roc_auc_score(y_test, class_proba_df[1])
print('Using the predictions, the area under the ROC curve is: %.3f' % auc_score)

# using the trapezoidal rule
auc_trapez_score = auc(fpr, tpr)
print('Using the trapezoidal rule, the area under the ROC curve is: %.3f' % auc_trapez_score)

# plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

pdb.set_trace()

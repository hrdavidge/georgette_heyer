import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages of age of readers introduced")
parser.add_argument('file',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--file',dest='file',default=None)
parser.add_argument('save',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--save',dest='save',default=None)

args = parser.parse_args()
file_path = args.file
save_path = args.save

if file_path == None:
    quit("User must specify the path to the survey csv")
if save_path == None:
    quit("User must specify the path to the folder of where to save the output graphs")

# read in csv
df = pd.read_csv(file_path)

# remove the NaNs from df['age_first_read']
df = df[df['age_first_read'].notnull()]

# find all the unique values of age_first_read
age_first_read_unique = df['age_first_read'].unique()
# convert to int
age_first_read_unique = age_first_read_unique.astype(int)
# order
age_first_read_ordered = np.sort(age_first_read_unique)

print 'Unique ages first read: ', age_first_read_ordered

# bin the data in column age_first_read into decades
# create a new age column
df['new_age'] = 0

# set all new_age columns to numbers
df.loc[(df['age_first_read'] >= 0) & (df['age_first_read'] < 10), ['new_age']] = int(0)
df.loc[(df['age_first_read'] >= 10) & (df['age_first_read'] < 20), ['new_age']] = int(10)
df.loc[(df['age_first_read'] >= 20) & (df['age_first_read'] < 30), ['new_age']] = int(20)
df.loc[(df['age_first_read'] >= 30) & (df['age_first_read'] < 40), ['new_age']] = int(30)
df.loc[(df['age_first_read'] >= 40) & (df['age_first_read'] < 50), ['new_age']] = int(40)
df.loc[(df['age_first_read'] >= 50) & (df['age_first_read'] < 60), ['new_age']] = int(60)

# plot frequency distribution
plt.figure()
plt.hist(df['new_age'], bins = 6)
plt.title('Age Readers Introduced to Heyer')
plt.xlabel('Age / Years')
plt.ylabel('Frequency')
plt.savefig(os.path.join(save_path + 'age_of_readers_introduced_freq.png'))

# save the percentages as a csv
under_ten = float(len(df.loc[df['new_age'] == 0])) / len(df) * 100.0
teens = float(len(df.loc[df['new_age'] == 10])) / len(df) * 100.0
twenties = float(len(df.loc[df['new_age'] == 20])) / len(df) * 100.0
thirties = float(len(df.loc[df['new_age'] == 30])) / len(df) * 100.0
forties = float(len(df.loc[df['new_age'] == 40])) / len(df) * 100.0
fifties = float(len(df.loc[df['new_age'] == 60])) / len(df) * 100.0

age = [ 'under_ten', 'teens', 'twenties', 'thirties', 'forties', 'fifties']
percent = [under_ten, teens, twenties, thirties, forties, fifties]

percent_df = pd.DataFrame({'age': age})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'age_of_readers_introduced_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
labels = ['Under Tens', 'Teens', 'Twenties', 'Thirties', 'Forties', 'Fifties']
sizes = [under_ten, teens, twenties, thirties, forties, fifties]
colors = ['red', 'darkorchid', 'slategrey', 'green', 'yellow', 'dodgerblue']
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%', colors=colors)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'age_of_readers_introduced_pie.png'))

print('Finished!')
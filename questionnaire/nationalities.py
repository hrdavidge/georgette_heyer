import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' nationalities")
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

# remove the NaNs from df['nationality1']
df = df[df['nationality1'].notnull()]
df = df[df['nationality2'].notnull()]
df = df[df['nationality3'].notnull()]

# get the unique values from all three columns
all_nationalities_unique_df = np.unique(df[['nationality1', 'nationality2', 'nationality3']].values)

# find all the unique values of introduced
nationality_unique_temp = df['nationality1'].unique()

# let's rename all English and Scottish to British
df.loc[df['nationality1'] == 'English', ['nationality1']] = 'British'
df.loc[df['nationality1'] == 'Scottish', ['nationality1']] = 'British'

# find all the unique values of introduced
nationality_unique = df['nationality1'].unique()

plt.figure()
df['nationality1'].value_counts().plot(kind='bar', title = "Nationalities of Heyer's Readers", y = 'Frequency')
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'nationalities_freq.png'))


# save the percentages as a csv
# create a new column
df['new_nationality1'] = 0

# loop through country_unique to find number in each
for num in range(0, len(nationality_unique)):
    df.loc[df['nationality1'] == nationality_unique[num], ['new_nationality1']] = int(num)

# save the percentages as a csv
zero = float(len(df.loc[df['new_nationality1'] == 0])) / len(df) * 100.0
one = float(len(df.loc[df['new_nationality1'] == 1])) / len(df) * 100.0
two = float(len(df.loc[df['new_nationality1'] == 2])) / len(df) * 100.0
three = float(len(df.loc[df['new_nationality1'] == 3])) / len(df) * 100.0
four = float(len(df.loc[df['new_nationality1'] == 4])) / len(df) * 100.0
five = float(len(df.loc[df['new_nationality1'] == 5])) / len(df) * 100.0
six = float(len(df.loc[df['new_nationality1'] == 6])) / len(df) * 100.0
seven = float(len(df.loc[df['new_nationality1'] == 7])) / len(df) * 100.0
eight = float(len(df.loc[df['new_nationality1'] == 8])) / len(df) * 100.0
nine = float(len(df.loc[df['new_nationality1'] == 9])) / len(df) * 100.0
ten = float(len(df.loc[df['new_nationality1'] == 10])) / len(df) * 100.0
eleven = float(len(df.loc[df['new_nationality1'] == 11])) / len(df) * 100.0
twelve = float(len(df.loc[df['new_nationality1'] == 12])) / len(df) * 100.0
thirteen = float(len(df.loc[df['new_nationality1'] == 13])) / len(df) * 100.0
fourteen = float(len(df.loc[df['new_nationality1'] == 14])) / len(df) * 100.0
fifteen = float(len(df.loc[df['new_nationality1'] == 15])) / len(df) * 100.0
sixteen = float(len(df.loc[df['new_nationality1'] == 16])) / len(df) * 100.0
seventeen = float(len(df.loc[df['new_nationality1'] == 17])) / len(df) * 100.0

percent = [zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen,\
           sixteen, seventeen]
labels = nationality_unique
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'nationalities_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = percent
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'nationalities_pie.png'))

print('Finished!')
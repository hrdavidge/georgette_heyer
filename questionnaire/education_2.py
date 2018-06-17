import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' equalifications")
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
df = df[df['education'].notnull()]

# find all the unique values of introduced
education_unique_temp = df['education'].unique()

# let's rename all English and Scottish to British
df.loc[df['education'] == 'profession qualification', ['education']] = 'Degree'
df.loc[df['education'] == 'PhD', ['education']] = 'Degree'
df.loc[df['education'] == 'masters', ['education']] = 'Degree'
df.loc[df['education'] == 'postgrad study', ['education']] = 'Degree'
df.loc[df['education'] == 'Masters', ['education']] = 'Degree'
df.loc[df['education'] == 'Postgrad study', ['education']] = 'Degree'
df.loc[df['education'] == 'degree', ['education']] = 'Degree'
df.loc[df['education'] == 'Degree', ['education']] = 'Degree'
df.loc[df['education'] == 'degree ongoing', ['education']] = 'Degree'
df.loc[df['education'] == 'de', ['education']] = 'Degree'
df.loc[df['education'] == 'foundation degree', ['education']] = 'No Degree'
df.loc[df['education'] == 'A levels', ['education']] = 'No Degree'
df.loc[df['education'] == 'a levels', ['education']] = 'No Degree'
df.loc[df['education'] == 'GCSE', ['education']] = 'No Degree'

# find all the unique values of introduced
education_unique = df['education'].unique()

# save frequency plot
plt.figure()
df['education'].value_counts().plot(kind='bar', title = "Degree Level of Heyer's Readers", y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'degree_freq.png'))

# save the percentages as a csv
zero = float(len(df.loc[df['education'] == 'Degree'])) / len(df) * 100.0
one = float(len(df.loc[df['education'] == 'No Degree'])) / len(df) * 100.0


percent = [zero, one]
labels = ['Degree', 'No Degree']
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'degree_percent.csv'), index=False)

# create and save a pit chart of the data
labels = [str('Degree '+ str(int(round(zero)))+'%'), str('No Degree '+ str(int(round(one)))+'%')]

plt.figure()
sizes = percent
colors = ['red', 'dodgerblue']
patches, texts, autotexts = plt.pie(sizes, startangle=90, colors=colors, autopct='%.0f%%')#shadow=True, , autotexts
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'degree_pie.png'))

print('Finished!')
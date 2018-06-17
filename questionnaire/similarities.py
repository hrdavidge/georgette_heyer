import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' similarities to Heyer")
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

# remove the NaNs from df['similarity_heyer']
df = df[df['similarity_heyer'].notnull()]

# find all the unique values of introduced
similarities_unique = df['similarity_heyer'].unique()

# rename the entries in column similarity_heyer
df.loc[df['similarity_heyer'] == 'Some similarities', ['similarity_heyer']] = 'Some Similarities'
df.loc[df['similarity_heyer'] == 'No similarities', ['similarity_heyer']] = 'No Similarities'
df.loc[df['similarity_heyer'] == 'Have not read a biography on her', ['similarity_heyer']] = 'Not Read Biography'
df.loc[df['similarity_heyer'] == 'have not read a biography on her', ['similarity_heyer']] = 'Not Read Biography'

plt.figure()
df['similarity_heyer'].value_counts().plot(kind='bar', title = "Readers' Similarities to Heyer", y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'similarities_freq.png'))

# save the percentages as a csv
some = float(len(df.loc[df['similarity_heyer'] == 'Some Similarities'])) / len(df) * 100.0
no = float(len(df.loc[df['similarity_heyer'] == 'No Similarities'])) / len(df) * 100.0
not_read = float(len(df.loc[df['similarity_heyer'] == 'Not Read Biography'])) / len(df) * 100.0

percent = [some, no, not_read]
labels = ['Some Similarities', 'No Similarities', 'Not Read Biography']
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'similarities_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = [some, no, not_read]
colors = ['lightskyblue', 'aquamarine', 'lightpink']
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%', colors=colors)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'similarities_pie.png'))

print ('Finished!')
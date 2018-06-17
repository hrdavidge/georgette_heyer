import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing if readers went thourhg mainstream education or not")
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

# remove the NaNs from df['mainstream']
df = df[df['mainstream'].notnull()]

# rename the entries in column similarity_heyer
df.loc[df['mainstream'] == 0, ['mainstream']] = 'Had a Mainstream Education'
df.loc[df['mainstream'] == 1, ['mainstream']] = 'Did not have a Mainstream Education'
df.loc[df['mainstream'] == 3, ['mainstream']] = 'Did not have a Mainstream Education'

plt.figure()
df['mainstream'].value_counts().plot(kind='bar', title = "Mainstream Eductions of Heyer's readers", y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'mainstream.png'))

# save the percentages as a csv
yes = float(len(df.loc[df['mainstream'] == 'Had a Mainstream Education'])) / len(df) * 100.0
no = float(len(df.loc[df['mainstream'] == 'Did not have a Mainstream Education'])) / len(df) * 100.0

percent = [yes, no]
labels = ['Had a Mainstream Education', 'Did not have a Mainstream Education']
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'mainstream.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = [yes, no]
colors = ['lightskyblue', 'lightpink']
labels = [str('Had a Mainstream Education ' + str(int(round(yes))) + '%'), str('Did not have a Mainstream Education ' + str(int(round(no))) + '%')]
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%', colors=colors)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'mainstream_pie.png'))

print ('Finished!')
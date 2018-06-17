import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing if readers identify as feminist or not")
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

# remove the NaNs from df['feminist']
df = df[df['feminist'].notnull()]

# convert numbers in ['feminist'] to words
df.loc[df['feminist'] == 0, ['feminist']] = str('Not a feminist')
df.loc[df['feminist'] == 1, ['feminist']] = str('Uncertain')
df.loc[df['feminist'] == 2, ['feminist']] = str('Feminist ally')
df.loc[df['feminist'] == 3, ['feminist']] = str('Feminist')
df.loc[df['feminist'] == 4, ['feminist']] = str('Feminist')

plt.figure()
df['feminist'].value_counts().plot(kind='bar', title = 'Whether Readers Identify as Feminist or not', y = 'Frequency')
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(save_path + 'feminist_freq.png'))

# save the percentages as a csv
nott = float(len(df.loc[df['feminist'] == 'Not a feminist'])) / len(df) * 100.0
uncertain = float(len(df.loc[df['feminist'] == 'Uncertain'])) / len(df) * 100.0
ally = float(len(df.loc[df['feminist'] == 'Feminist ally'])) / len(df) * 100.0
yes = float(len(df.loc[df['feminist'] == 'Feminist'])) / len(df) * 100.0

percent = [nott, uncertain, ally, yes]
labels = ['Not a feminist', 'Uncertain', 'Feminist ally', 'Feminist']
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'feminist_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = [nott, uncertain, ally, yes]
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'feminist_pie.png'))

print('Finished!')
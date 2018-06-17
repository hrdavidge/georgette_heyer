import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' english equalifications")
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

# remove the NaNs from df['qual_eng']
df = df[df['qual_eng'].notnull()]

# find all the unique values of english qualification
education_unique_temp = df['qual_eng'].unique()

# let's rename all English and Scottish to British
df.loc[df['qual_eng'] == 'PhD', ['qual_eng']] = 'Level 8'
df.loc[df['qual_eng'] == 'masters', ['qual_eng']] = 'Level 7'
df.loc[df['qual_eng'] == 'Postgrad study', ['qual_eng']] = 'Level 7'
df.loc[df['qual_eng'] == 'Masters', ['qual_eng']] = 'Level 7'
df.loc[df['qual_eng'] == 'Degree', ['qual_eng']] = 'Level 6'
df.loc[df['qual_eng'] == 'Degree unfinished', ['qual_eng']] = 'Level 6'
df.loc[df['qual_eng'] == 'foundation degree', ['qual_eng']] = 'Level 5'
df.loc[df['qual_eng'] == 'A level', ['qual_eng']] = 'Level 3'
df.loc[df['qual_eng'] == 'GCSE', ['qual_eng']] = 'Level 2'
df.loc[df['qual_eng'] == 'no', ['qual_eng']] = 'None'

# find all the unique values of histry education
education_unique = df['qual_eng'].unique()

# save frequency plot
plt.figure()
df['qual_eng'].value_counts().plot(kind='bar', title = "English Education Level of Heyer's Readers", y = 'Frequency', sort_columns = ['None', 'Level 2', 'Level 3', 'Level 5', 'Level 6', 'Level 7', 'Level 8'])#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'english_education_freq.png'))

l8 = len(df.loc[df['qual_eng'] == 'Level 8'])
l7 = len(df.loc[df['qual_eng'] == 'Level 7'])
l6 = len(df.loc[df['qual_eng'] == 'Level 6'])
l5 = len(df.loc[df['qual_eng'] == 'Level 5'])
l3 = len(df.loc[df['qual_eng'] == 'Level 3'])
l2 = len(df.loc[df['qual_eng'] == 'Level 2'])
noo = len(df.loc[df['qual_eng'] == 'None'])

labels = ['None', 'Level 2', 'Level 3', 'Level 5', 'Level 6', 'Level 7', 'Level 8']

# save the percentages as a csv
zero = float(len(df.loc[df['qual_eng'] == 'Level 8'])) / len(df) * 100.0
one = float(len(df.loc[df['qual_eng'] == 'Level 7'])) / len(df) * 100.0
two = float(len(df.loc[df['qual_eng'] == 'Level 6'])) / len(df) * 100.0
three = float(len(df.loc[df['qual_eng'] == 'Level 5'])) / len(df) * 100.0
four = float(len(df.loc[df['qual_eng'] == 'Level 3'])) / len(df) * 100.0
five = float(len(df.loc[df['qual_eng'] == 'Level 2'])) / len(df) * 100.0
six = float(len(df.loc[df['qual_eng'] == 'None'])) / len(df) * 100.0

percent = [six, five, four, three, two, one, zero]
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'english_education_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = percent
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'rebeccapurple', 'darkviolet']
patches, texts, autotexts = plt.pie(sizes, startangle=90, colors=colors, autopct='%.0f%%', )#shadow=True,
plt.legend(patches, labels, loc="lower left")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'english_education_2_pie.png'))

labels = [str('None '+ str(int(round(six)))+'%'), str('Level 2 '+ str(int(round(five)))+'%'), str('Level 3 '+ str(int(round(four)))+'%'), str('Level 5 '+ str(int(round(three)))+'%'), str('Level 6 '+ str(int(round(two)))+'%'), str('Level 7 '+ str(int(round(one)))+'%'), str('Level 8 '+ str(int(round(zero)))+'%')]

plt.figure()
sizes = percent
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'rebeccapurple', 'darkviolet']
patches, texts = plt.pie(sizes, startangle=90, colors=colors)#autopct='%.0f%%', )#shadow=True, , autotexts
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'english_education_1_pie.png'))

print('Finished!')
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

# remove the NaNs from df['education']
df = df[df['education'].notnull()]

# find all the unique values of introduced
education_unique_temp = df['education'].unique()

# let's rename all English and Scottish to British
df.loc[df['education'] == 'profession qualification', ['education']] = 'Professional'
df.loc[df['education'] == 'PhD', ['education']] = 'Level 8'
df.loc[df['education'] == 'masters', ['education']] = 'Level 7'
df.loc[df['education'] == 'postgrad study', ['education']] = 'Level 7'
df.loc[df['education'] == 'Masters', ['education']] = 'Level 7'
df.loc[df['education'] == 'Postgrad study', ['education']] = 'Level 7'
df.loc[df['education'] == 'degree', ['education']] = 'Level 6'
df.loc[df['education'] == 'Degree', ['education']] = 'Level 6'
df.loc[df['education'] == 'degree ongoing', ['education']] = 'Level 6'
df.loc[df['education'] == 'de', ['education']] = 'Level 6'
df.loc[df['education'] == 'foundation degree', ['education']] = 'Level 5'
df.loc[df['education'] == 'A levels', ['education']] = 'Level 3'
df.loc[df['education'] == 'a levels', ['education']] = 'Level 3'
df.loc[df['education'] == 'GCSE', ['education']] = 'Level 2'

# find all the unique values of introduced
education_unique = df['education'].unique()

# save frequency plot
plt.figure()
df['education'].value_counts().plot(kind='bar', title = "Education Level of Heyer's Readers", y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'education_freq.png'))

# save the percentages as a csv
zero = float(len(df.loc[df['education'] == 'Professional'])) / len(df) * 100.0
one = float(len(df.loc[df['education'] == 'Level 8'])) / len(df) * 100.0
two = float(len(df.loc[df['education'] == 'Level 7'])) / len(df) * 100.0
three = float(len(df.loc[df['education'] == 'Level 6'])) / len(df) * 100.0
four = float(len(df.loc[df['education'] == 'Level 5'])) / len(df) * 100.0
five = float(len(df.loc[df['education'] == 'Level 3'])) / len(df) * 100.0
six = float(len(df.loc[df['education'] == 'Level 2'])) / len(df) * 100.0

percent = [zero, one, two, three, four, five, six]
labels = ['Professional', 'Level 8', 'Level 7', 'Level 6', 'Level 5', 'Level 3', 'Level 2']
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'education_percent.csv'), index=False)

# create and save a pit chart of the data
labels = [str('Professional '+ str(int(round(zero)))+'%'), str('Level 8 '+ str(int(round(one)))+'%'), str('Level 7 '+ str(int(round(two)))+'%'), str('Level 6 '+ str(int(round(three)))+'%'), str('Level 5 '+ str(int(round(four)))+'%'), str('Level 3 '+ str(int(round(five)))+'%'), str('Level 2 '+ str(int(round(six)))+'%')]

plt.figure()
sizes = percent
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'rebeccapurple', 'darkviolet']
patches, texts = plt.pie(sizes, startangle=90, colors=colors)#autopct='%.0f%%', )#shadow=True, , autotexts
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'education_pie.png'))

print('Finished!')
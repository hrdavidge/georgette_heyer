import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing who introduced reader to Heyer")
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

# remove the NaNs from df['introduced']
df = df[df['introduced'].notnull()]

# find all the unique values of introduced
introduced_unique = df['introduced'].unique()

print 'Unique list of ways people were introduced to Heyer: ', introduced_unique

plt.figure()
df['introduced'].value_counts().plot(kind='bar', title = 'The ways readers were introduced to Heyer', x = 'Way Introduced', y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'way_readers_introduced_freq.png'))

# create a new age column
df['new_introduce'] = 0

# loop through the all the unique ways introduced
for num in range(0, len(introduced_unique)):
    df.loc[df['introduced'] == introduced_unique[num], ['new_introduce']] = int(num)

# save the percentages as a csv
zero = float(len(df.loc[df['new_introduce'] == 0])) / len(df) * 100.0
one = float(len(df.loc[df['new_introduce'] == 1])) / len(df) * 100.0
two = float(len(df.loc[df['new_introduce'] == 2])) / len(df) * 100.0
three = float(len(df.loc[df['new_introduce'] == 3])) / len(df) * 100.0
four = float(len(df.loc[df['new_introduce'] == 4])) / len(df) * 100.0
five = float(len(df.loc[df['new_introduce'] == 5])) / len(df) * 100.0
six = float(len(df.loc[df['new_introduce'] == 6])) / len(df) * 100.0
seven = float(len(df.loc[df['new_introduce'] == 7])) / len(df) * 100.0
eight = float(len(df.loc[df['new_introduce'] == 8])) / len(df) * 100.0
nine = float(len(df.loc[df['new_introduce'] == 9])) / len(df) * 100.0
ten = float(len(df.loc[df['new_introduce'] == 10])) / len(df) * 100.0
eleven = float(len(df.loc[df['new_introduce'] == 11])) / len(df) * 100.0
twelve = float(len(df.loc[df['new_introduce'] == 12])) / len(df) * 100.0
thirteen = float(len(df.loc[df['new_introduce'] == 13])) / len(df) * 100.0
fourteen = float(len(df.loc[df['new_introduce'] == 14])) / len(df) * 100.0
fifteen = float(len(df.loc[df['new_introduce'] == 15])) / len(df) * 100.0
sixteen = float(len(df.loc[df['new_introduce'] == 16])) / len(df) * 100.0
seventeen = float(len(df.loc[df['new_introduce'] == 17])) / len(df) * 100.0
eighteen = float(len(df.loc[df['new_introduce'] == 18])) / len(df) * 100.0
nineteen = float(len(df.loc[df['new_introduce'] == 19])) / len(df) * 100.0
twenty = float(len(df.loc[df['new_introduce'] == 20])) / len(df) * 100.0
twentyone = float(len(df.loc[df['new_introduce'] == 21])) / len(df) * 100.0
twentytwo = float(len(df.loc[df['new_introduce'] == 22])) / len(df) * 100.0
twentythree = float(len(df.loc[df['new_introduce'] == 23])) / len(df) * 100.0
twentyfour = float(len(df.loc[df['new_introduce'] == 24])) / len(df) * 100.0
twentyfive = float(len(df.loc[df['new_introduce'] == 25])) / len(df) * 100.0


percent = [zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen,\
           sixteen, seventeen, eighteen, nineteen, twenty, twentyone, twentytwo,  twentythree, twentyfour, twentyfive]

percent_df = pd.DataFrame({'method': introduced_unique})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'way_readers_introduced_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
labels = introduced_unique
sizes = [zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen,\
           sixteen, seventeen, eighteen, nineteen, twenty, twentyone, twentytwo,  twentythree, twentyfour, twentyfive]
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'way_readers_introduced_pie.png'))

print ('Finished!')
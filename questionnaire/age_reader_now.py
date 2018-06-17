import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages of age of reader now")
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

# remove the NaNs from df['age']
df = df[df['age'].notnull()]

# create a new age column
df['new_age'] = df['age']

# set all new_age columns to numbers
df.loc[df['new_age'] == '20 - 29', ['new_age']] = int(20)
df.loc[df['new_age'] == '30 - 39', ['new_age']] = int(30)
df.loc[df['new_age'] == '40 - 49', ['new_age']] = int(40)
df.loc[df['new_age'] == '50 - 59', ['new_age']] = int(50)
df.loc[df['new_age'] == '60 - 69', ['new_age']] = int(60)
df.loc[df['new_age'] == '70 -79', ['new_age']] = int(70)
df.loc[df['new_age'] == '80 - 89', ['new_age']] = int(90)

# plot frequency distribution
plt.hist(df['new_age'], bins = 7)
plt.title('Age of Readers')
plt.xlabel('Age / Years')
plt.ylabel('Frequency')
plt.savefig(os.path.join(save_path + 'age_of_readers_now_freq.png'))

# save the percentages as a csv
twenties = float(len(df.loc[df['new_age'] == 20])) / len(df) * 100.0
thirties = float(len(df.loc[df['new_age'] == 30])) / len(df) * 100.0
forties = float(len(df.loc[df['new_age'] == 40])) / len(df) * 100.0
fifties = float(len(df.loc[df['new_age'] == 50])) / len(df) * 100.0
sixties = float(len(df.loc[df['new_age'] == 60])) / len(df) * 100.0
seventies = float(len(df.loc[df['new_age'] == 70])) / len(df) * 100.0
eighties = float(len(df.loc[df['new_age'] == 90])) / len(df) * 100.0

age = ['twenties', 'thirties', 'forties', 'fifties', 'sixties', 'seventies', 'eighties']
percent = [twenties, thirties, forties, fifties, sixties, seventies, eighties]

percent_df = pd.DataFrame({'age': age})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'age_of_readers_now_percent.csv'), index=False)

# create and save a pit chart of the data
labels = ['Twenties', 'Thirties', 'Forties', 'Fifties', 'Sixties', 'Seventies', 'Eighties']
sizes = [twenties, thirties, forties, fifties, sixties, seventies, eighties]
colors = ['red', 'darkorange', 'yellow', 'green', 'blue', 'rebeccapurple', 'darkviolet']
patches, texts, autotexts = plt.pie(sizes, startangle=90, colors=colors, autopct='%.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'age_of_readers_now_pie.png'))

print('Finished!')
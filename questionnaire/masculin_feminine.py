import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to read in novels dataframe, and create a scatter plot of how masculin and feminine Heyer's books are")
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

# remove rows which don't have a m/f entry
# remove the NaNs from df['male']
df = df[df['male'].notnull()]

# create a temp df, removing the three lower outliers
temp_df = df.drop([33, 40, 46])

# convert pecentage_female to float
temp_df['percentage_female'] = temp_df['percentage_female'].astype(float)

# create the scatter plots
# show complete percentage of female to male pronouns
plt.figure()
plt.plot(df['pub'], df['percentage_female'], 'o', marker = '*', color = 'red', label='Novels')
plt.plot(temp_df['pub'], np.poly1d(np.polyfit(temp_df['pub'], temp_df['percentage_female'], 1))(temp_df['pub']), color='black', label='Line showing increase')
plt.plot([1962.5, 1962.5], [-10, 110], color = 'blue', label="Year Heyer's son married")
plt.xlabel('Year of Publication')
plt.ylabel('Percentage of Female Pronouns Used')
plt.title("Percentage of Female to Male Description in Heyer's Novels")
plt.grid()
plt.legend(loc="upper left")
plt.axis((1920, 1975, 10, 70))
plt.savefig(os.path.join(save_path + 'female_pronouns_percent.png'))

# show she/he and her/him,his percentage separately
plt.figure()
plt.plot(df['pub'], df['percentage_of_she'], 'o', marker = '*', color = 'red', label = 'Percent of She')
plt.plot(df['pub'], df['percentage_of_her'], 'o', marker = 'x', color = 'blue', label = 'Percent of Her')
plt.xlabel('Year of Publication')
plt.ylabel('Percentage of Female Pronouns Used')
plt.title("Percentage of Female to Male Description in Heyer's Novels")
plt.legend(loc="upper left")
plt.grid()
plt.savefig(os.path.join(save_path + 'she_her_percent.png'))

print('Finished!')
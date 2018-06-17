import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' locations")
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

# remove the NaNs from df['country1']
df = df[df['country1'].notnull()]
df = df[df['country2'].notnull()]
df = df[df['country3'].notnull()]

# get the unique values from all three columns
all_counties_unique_df = np.unique(df[['country1', 'country2', 'country3']].values)

# find all the unique values of introduced
country_unique_temp = df['country1'].unique()

# let's rename all England, Scotland, Northen Ireland, Welsh, UK to Britain
df.loc[df['country1'] == 'UK', ['country1']] = 'Britain'
df.loc[df['country1'] == 'Scotland', ['country1']] = 'Britain'
df.loc[df['country1'] == 'Wales', ['country1']] = 'Britain'
df.loc[df['country1'] == 'England', ['country1']] = 'Britain'
df.loc[df['country1'] == 'Northern Ireland', ['country1']] = 'Britain'

# find all the unique values of introduced
country_unique = df['country1'].unique()

plt.figure()
df['country1'].value_counts().plot(kind='bar', title = "Countries Heyer's Readers Live In", y = 'Frequency')#
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'countries_freq.png'))

# save the percentages as a csv
# create a new column
df['new_country1'] = 0

# loop through country_unique to find number in each
for num in range(0, len(country_unique)):
    df.loc[df['country1'] == country_unique[num], ['new_country1']] = int(num)

# save the percentages as a csv
zero = float(len(df.loc[df['new_country1'] == 0])) / len(df) * 100.0
one = float(len(df.loc[df['new_country1'] == 1])) / len(df) * 100.0
two = float(len(df.loc[df['new_country1'] == 2])) / len(df) * 100.0
three = float(len(df.loc[df['new_country1'] == 3])) / len(df) * 100.0
four = float(len(df.loc[df['new_country1'] == 4])) / len(df) * 100.0
five = float(len(df.loc[df['new_country1'] == 5])) / len(df) * 100.0
six = float(len(df.loc[df['new_country1'] == 6])) / len(df) * 100.0
seven = float(len(df.loc[df['new_country1'] == 7])) / len(df) * 100.0
eight = float(len(df.loc[df['new_country1'] == 8])) / len(df) * 100.0
nine = float(len(df.loc[df['new_country1'] == 9])) / len(df) * 100.0
ten = float(len(df.loc[df['new_country1'] == 10])) / len(df) * 100.0
eleven = float(len(df.loc[df['new_country1'] == 11])) / len(df) * 100.0
twelve = float(len(df.loc[df['new_country1'] == 12])) / len(df) * 100.0
thirteen = float(len(df.loc[df['new_country1'] == 13])) / len(df) * 100.0
fourteen = float(len(df.loc[df['new_country1'] == 14])) / len(df) * 100.0
fifteen = float(len(df.loc[df['new_country1'] == 15])) / len(df) * 100.0
sixteen = float(len(df.loc[df['new_country1'] == 16])) / len(df) * 100.0
seventeen = float(len(df.loc[df['new_country1'] == 17])) / len(df) * 100.0
eighteen = float(len(df.loc[df['new_country1'] == 18])) / len(df) * 100.0
nineteen = float(len(df.loc[df['new_country1'] == 19])) / len(df) * 100.0
twenty = float(len(df.loc[df['new_country1'] == 20])) / len(df) * 100.0
twentyone = float(len(df.loc[df['new_country1'] == 21])) / len(df) * 100.0

percent = [zero, one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen,\
           sixteen, seventeen, eighteen, nineteen, twenty, twentyone]
labels = country_unique
percent_df = pd.DataFrame({'method': labels})
percent_df['percent'] = percent

percent_df.to_csv(os.path.join(save_path + 'countries_percent.csv'), index=False)

# create and save a pit chart of the data
plt.figure()
sizes = percent
patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'countries_pie.png'))

# try combining some countries, and creating a second pie chart
# renaming the countries
df.loc[df['country1'] == 'Pakistan', ['country1']] = 'Asia'
df.loc[df['country1'] == 'Sri Lanka', ['country1']] = 'Asia'
df.loc[df['country1'] == 'Republic of Korea', ['country1']] = 'Asia'
df.loc[df['country1'] == 'Singapore', ['country1']] = 'Asia'
df.loc[df['country1'] == 'India', ['country1']] = 'Asia'
df.loc[df['country1'] == 'Egypt', ['country1']] = 'Middle East'
df.loc[df['country1'] == 'United Arab Emirates', ['country1']] = 'Middle East'
df.loc[df['country1'] == 'Italy', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Finland', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Romania', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Sweden', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Poland', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'France', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Denmark', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Germany', ['country1']] = 'Continental Europe'
df.loc[df['country1'] == 'Norway', ['country1']] = 'Continental Europe'

# find all the unique values of introduced
country_unique = df['country1'].unique()

# just to check all the countries are as I want them
plt.figure()
df['country1'].value_counts().plot(kind='bar', title = "Countries Heyer's Readers Live In", y = 'Frequency')#
plt.tight_layout()
plt.show()


zero = float(len(df.loc[df['country1'] == 'America'])) / len(df) * 100.0
one = float(len(df.loc[df['country1'] == 'Britain'])) / len(df) * 100.0
two = float(len(df.loc[df['country1'] == 'Australia'])) / len(df) * 100.0
three = float(len(df.loc[df['country1'] == 'Continental Europe'])) / len(df) * 100.0
four = float(len(df.loc[df['country1'] == 'Canada'])) / len(df) * 100.0
five = float(len(df.loc[df['country1'] == 'Asia'])) / len(df) * 100.0
six = float(len(df.loc[df['country1'] == 'Ireland'])) / len(df) * 100.0
seven = float(len(df.loc[df['country1'] == 'New Zealand'])) / len(df) * 100.0
eight = float(len(df.loc[df['country1'] == 'Middle East'])) / len(df) * 100.0

percent = [zero, one, two, three, four, five, six, seven, eight]
labels = [str('America '+ str(int(round(zero)))+'%'), str('Britain '+ str(int(round(one)))+'%'), str('Australia '+ str(int(round(two)))+'%'), str('Continental Europe '+ str(int(round(three)))+'%'), str('Canada '+ str(int(round(four)))+'%'), str('Asia '+ str(int(round(five)))+'%'), str('Ireland '+ str(int(round(six)))+'%'), str('New Zealand '+ str(int(round(seven)))+'%'), str('Middle East '+ str(int(round(eight)))+'%')]

plt.figure()
sizes = percent
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(patches, labels, loc="upper left")
plt.axis('equal')
plt.savefig(os.path.join(save_path + 'countries2_pie.png'))

print('Finished!')
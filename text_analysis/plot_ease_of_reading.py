import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser("Code to plot the ease of reading for of Heyer's novels")
parser.add_argument('file',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--file',dest='file',default=None)
parser.add_argument('save',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--save',dest='save',default=None)

args = parser.parse_args()
file_path = args.file
save_path = args.save

if file_path == None:
    quit("User must specify the path to the csv of ease of reading")
if save_path == None:
    quit("User must specify the path to the folder of where to save the output graph")

# read in csv
df = pd.read_csv(file_path)

# order csv by year
ordered_df = df.sort_values(by = ['year'])

# create an array the same size as the number of novels
novel_count = range(1, len(ordered_df) + 1)

# plot ease of reading chronologically
plt.figure()
plt.plot(df['year'], df['Flesch'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading ease / %')
plt.title('Measured using number of syllables and sentence length')
plt.suptitle('The Flesch Reading Ease Score')
plt.savefig(os.path.join(save_path + 'Flesch.png'))

plt.figure()
plt.plot(df['year'], df['Flesch_Kincaid'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using number of syllables and sentence length')
plt.suptitle('The Flesch-Kincaid Grade Level')
plt.savefig(os.path.join(save_path + 'Flesch_Kincaid.png'))

plt.figure()
plt.plot(df['year'], df['FOG'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using number of syllables and sentence length')
plt.suptitle('The Fog Scale (Gunning FOG Formula)')
plt.savefig(os.path.join(save_path + 'FOG.png'))

plt.figure()
plt.plot(df['year'], df['SMOG'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using number of syllables and sentence length')
plt.suptitle('The SMOG Index')
plt.savefig(os.path.join(save_path + 'SMOG.png'))

plt.figure()
plt.plot(df['year'], df['ARI'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using word length and sentence length')
plt.suptitle('The Automated Readability Index')
plt.savefig(os.path.join(save_path + 'ARI.png'))

plt.figure()
plt.plot(df['year'], df['Coleman_Liau'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using word length')
plt.suptitle('The Coleman-Liau Formula')
plt.savefig(os.path.join(save_path + 'Coleman_Liau.png'))

plt.figure()
plt.plot(df['year'], df['Lisear'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Complexity of Language')
plt.suptitle('The Lisear Write Formula')
plt.savefig(os.path.join(save_path + 'Lisear.png'))

plt.figure()
plt.plot(df['year'], df['Dale_Chall'], 'o', marker = '*', color = 'blue')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using 3000 most common words')
plt.suptitle('The New Dale-Chall Formula')
plt.savefig(os.path.join(save_path + 'Dale_Chall.png'))

# add in lines of best fit for two graphs
# now split the df into two
ari_df1 = df.iloc[[1, 2, 3, 4, 5, 6, 7]]
ari_df2 = df.iloc[[10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33]]

plt.figure()
plt.plot(df['year'], df['ARI'], 'o', marker = '*', color = 'red', label='Novels')
plt.plot(ari_df1['year'], np.poly1d(np.polyfit(ari_df1['year'], ari_df1['ARI'], 1))(ari_df1['year']), color='blue', label='Early Novels')
plt.plot(ari_df2['year'], np.poly1d(np.polyfit(ari_df2['year'], ari_df2['ARI'], 1))(ari_df2['year']), color='green', label='Later Novels')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using word length and sentence length')
plt.suptitle('The Automated Readability Index')
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_path + 'ARI_with_bestfit_lines.png'))

#dale_chall_df = df.drop[[0, 1, 9, 15, 31]]
# now split the df into two
dc_df1 = df.iloc[[2, 3, 4, 5, 6, 7, 8, 10]]
dc_df2 = df.iloc[[11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33]]

plt.figure()
plt.plot(df['year'], df['Dale_Chall'], 'o', marker = '*', color = 'red', label='Novels')
plt.plot(dc_df1['year'], np.poly1d(np.polyfit(dc_df1['year'], dc_df1['Dale_Chall'], 1))(dc_df1['year']), color='blue', label='Early Novels')
plt.plot(dc_df2['year'], np.poly1d(np.polyfit(dc_df2['year'], dc_df2['Dale_Chall'], 1))(dc_df2['year']), color='green', label='Later Novels')
plt.xlabel('Year Novel Published')
plt.ylabel('Reading Complexity / US school grade')
plt.title('Measured using 3000 most common words')
plt.suptitle('The New Dale-Chall Formula')
plt.legend(loc="upper left")
plt.savefig(os.path.join(save_path + 'Dale_Chall_with_bestfit_lines.png'))

print ('Finished!')
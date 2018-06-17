import pandas as pd
import argparse
import numpy as np
import os
import string

parser = argparse.ArgumentParser("Code to read in a word frequency csv, and output it ready for word cloud")
parser.add_argument('file',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--file',dest='file',default=None)
parser.add_argument('save',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--save',dest='save',default=None)

args = parser.parse_args()
file_path = args.file
save_path = args.save

if file_path == None:
    quit("User must specify the path to the .txt file")
if save_path == None:
    quit("User must specify the path to the folder of where to save the output csv file")

df = pd.read_csv(file_path)

# we just want the first 50 rows
#df = df[0:50]
# find the number of words in the minimum word
len_df = len(df)

#min_num_words = df['count'][49]
min_num_words = df['count'][len_df - 1]
# divide all the words by min_num_words, round down to neareat integer
df['rounded_count'] = np.floor(df['count'] / min_num_words)

# save the words to a new csv
words = []
# loop through df
for row in range(0, len(df)):
    # find the frequence of the word in row
    num = int(df['rounded_count'][row])
    for times in range(0, num):
        words.append(df['word'][row])

# find name of file
book_name_with_ext = os.path.basename(file_path)
# need to remove the extension
book_name = string.split(book_name_with_ext, '.')[0]

# convert words to df
words_df = pd.DataFrame(data = words)

# save new df
words_df.to_csv(os.path.join(save_path, book_name + 'temp.csv'), sep = '\t', index=False)

print('Finished!')
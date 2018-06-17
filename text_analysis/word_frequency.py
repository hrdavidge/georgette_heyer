import pandas as pd
import argparse
import os
import re
from collections import Counter
import string
import itertools
import collections

parser = argparse.ArgumentParser("Code to count word frequency")
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

num_of_words = int(100)

# find name of file
book_name_with_ext = os.path.basename(file_path)
# need to remove the extension
book_name = string.split(book_name_with_ext, '.')[0]

# read in the text, one character at a time
with open(file_path) as f:
    passage = f.read()
# split text up into words
words = re.findall(r'\w+', passage)

cap_words = [word.upper() for word in words]

all_word_counts = Counter(cap_words)
word_counts = all_word_counts.most_common(num_of_words)

d = collections.OrderedDict(word_counts)
x = itertools.islice(d.items(), 0, num_of_words)

word_list = []
count_list = []

# load word and count into dataframe
for key, value in x:
    word_and_count_tuple = (key, value)
    word = str(word_and_count_tuple[0])
    word_list.append(word)
    count_list.append(str(word_and_count_tuple[1]))
    print (key, value)

# create an index
index = range(0, num_of_words)

# save word and count into a dataframe
data = {'index': index, 'word': word_list, 'count': count_list}
df = pd.DataFrame(data = data, columns = ['index', 'word', 'count'])
df.to_csv(os.path.join(save_path, book_name + '_num_of_words_' + str(len(cap_words)) + '.csv'), index=False)

print ('Finished!')
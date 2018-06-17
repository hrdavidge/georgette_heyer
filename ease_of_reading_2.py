import pandas as pd
import argparse
import os
import string
from textstat.textstat import textstat

parser = argparse.ArgumentParser("Code to measure ease of reading for all of Heyer's novels")
parser.add_argument('file',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--file',dest='file',default=None)
parser.add_argument('save',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--save',dest='save',default=None)

args = parser.parse_args()
file_path = args.file
save_path = args.save

if file_path == None:
    quit("User must specify the path to the folder containing all the .txt file")
if save_path == None:
    quit("User must specify the path to the folder of where to save the output csv file")

# create a list of all of the files in file_path
files = os.listdir(file_path)

# create the dataframe to hold the results in
df = pd.DataFrame(columns = ['name', 'Flesch', 'Flesch_Kincaid', 'FOG', 'SMOG', 'ARI', 'Coleman_Liau', 'Lisear', 'Dale_Chall'])

# loop through the files in the directory pass
# calculate each reading level and save to df
for num in range(0, len(files)):
    # remove the type of file from name
    book_name = string.split(files[num], '.')[0]
    print(book_name)
    # read the text in as one string
    file = open(os.path.join(file_path, files[num]), 'r')
    text = file.read().strip()
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\xc3\xa9', 'e').replace("\xe2\x80\x9d",                                                                           '"')
    text = text.replace("\xe2\x80\x9c", '"').replace('\xe2\x80\x94', '-').replace('\xc3\xa8', 'e').replace('\xc3\xa2', 'a')
    text = text.replace('\xe2\x80\x98', '').replace('\xc3\xb4', 'o').replace('\xc3\xa7', 'c').replace('\xc3\xaf', 'i')
    text = text.replace('\xc3\xaa', 'e').replace('\xc3\xbb', 'u').replace("\xe2\x80\x99", "'").replace('\xc3\xa0', 'a')
    text = text.replace('\xc5\x92', 'OE').replace('\xc5\x93', 'oe').replace('\xc3\xae', 'i').replace('\xc3\x80', 'A')
    text = text.replace('\xc3\xbc', 'u').replace('\xe2\x80\x93', '-').replace('\xc3\xb6', 'o').replace('\xc3\xa4', 'a')
    text = text.replace('\xc3\xb1', 'n').replace('\xc3\x89', 'E').replace('\xc3\x9c', 'U').replace('\xc2\xab', '')
    text = text.replace('\xc2\xbb', '').replace('\xc2\xa3', 'pounds ').replace('\xe2\x80\xa6', '...')
    text = text.replace('\xe2\x80\xa2', '').replace('\xc3\xab', 'e').replace('\xc3\xb9', 'u').replace('\xc3\xa1', 'a')
    text = text.replace('\xe0', '?').replace('\xe9', 'e').replace('\xea', ' ').replace('\xc2\xb0', '')
    text = text.replace('\xc3\xb3', 'o').replace('\xc3\x87', 'C')
    file.close()

    # now run the text through the different language level algorithms
    Flesch = textstat.flesch_reading_ease(text)
    Flesch_Kincaid = textstat.flesch_kincaid_grade(text)
    FOG = textstat.gunning_fog(text)
    SMOG = textstat.smog_index(text)
    ARI = textstat.automated_readability_index(text)
    Coleman_Liau = textstat.coleman_liau_index(text)
    Lisear = textstat.linsear_write_formula(text)
    Dale_Chall = textstat.dale_chall_readability_score(text)

    # load these into df
    new_row = pd.DataFrame([[book_name, Flesch, Flesch_Kincaid, FOG, SMOG, ARI, Coleman_Liau, Lisear, Dale_Chall]], columns = ['name', 'Flesch', 'Flesch_Kincaid', 'FOG', 'SMOG', 'ARI', 'Coleman_Liau', 'Lisear', 'Dale_Chall'])
    df = df.append(new_row)
# save the new df
df.to_csv(os.path.join(save_path + 'ease_of_reading.csv'), index=False)

print('Finished!')
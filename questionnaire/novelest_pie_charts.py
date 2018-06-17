import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create pie charts of writers")
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


def create_pie_chart(name, df, abriv, master_df_len):
    print 'creating pie charts for: ', name
    # remove the NaNs from df
    df = df[df[abriv].notnull()]
    # save the percentages of each type
    enjoyed = float(len(df.loc[df[abriv] == 4])) / len(df) * 100.0
    okay = float(len(df.loc[df[abriv] == 3])) / len(df) * 100.0
    not_liked = float(len(df.loc[df[abriv] == 2])) / len(df) * 100.0
    not_read = float(len(df.loc[df[abriv] == 1])) / len(df) * 100.0

    sizes = [enjoyed, okay, not_liked, not_read]
    labels = ['Enjoyed', 'Thought ok', 'Not Enjoyed', 'Not Read']

    # set colours
    if abriv == 'ja' or abriv == 'b' or abriv == 'fb' or abriv == 'eg' or abriv == 'ar':
        colors = ['blueviolet', 'mediumpurple', 'plum', 'thistle']
    if abriv == 'efb' or abriv == 'ac' or abriv == 'pgw':
        colors = ['darkmagenta', 'm', 'hotpink', 'pink']
    if abriv == 'mcb' or abriv == 'cartland' or abriv == 'mag' or abriv == 'darcy' or abriv == 'mi' or abriv == 'nl' or abriv == 'as':
        colors = ['firebrick', 'red', 'darkorange', 'darksalmon']
    if  abriv == 'cornwell' or abriv == 'csf' or abriv == 'po':
        colors = ['darkgreen', 'green', 'limegreen', 'lightgreen']
    if abriv == 'dg' or abriv == 'rh' or abriv == 'pg':
        colors = ['mediumblue', 'royalblue', 'dodgerblue', 'lightskyblue']
    if abriv == 'dickens' or abriv == 'ad' or abriv == 'th' or abriv == 'ws':
        colors = ['dimgrey', 'darkgrey', 'lightgrey', 'whitesmoke']

    # save figure with percentages
    plt.figure()
    patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%', colors=colors)
    plt.legend(patches, labels, loc="upper left")
    plt.axis('equal')
    plt.savefig(os.path.join(save_path + str(name+'_pie_percentages.png')))

    # save figure without percentages
    labels = [str('Enjoyed ' + str(int(round(enjoyed))) + '%'), str('Thought ok ' + str(int(round(okay))) + '%'), str('Not Enjoyed ' + str(int(round(not_liked))) + '%'), str('Not Read ' + str(int(round(not_read))) + '%')]
    plt.figure()
    patches, texts = plt.pie(sizes, startangle=90, colors=colors)
    plt.legend(patches, labels, loc="upper left")
    plt.axis('equal')
    plt.savefig(os.path.join(save_path + str(name + '_pie.png')))

    # now just looking at people who have read the novels
    # remove the people who have not read the novel
    df = df.loc[df[abriv] != 1]

    # save the percentages of each type
    enjoyed = float(len(df.loc[df[abriv] == 4])) / len(df) * 100.0
    okay = float(len(df.loc[df[abriv] == 3])) / len(df) * 100.0
    not_liked = float(len(df.loc[df[abriv] == 2])) / len(df) * 100.0

    sizes = [enjoyed, okay, not_liked]
    labels = ['Enjoyed', 'Thought ok', 'Not Enjoyed']

    # set colours
    if abriv == 'ja' or abriv == 'b' or abriv == 'fb' or abriv == 'eg' or abriv == 'ar':
        colors = ['blueviolet', 'mediumpurple', 'plum']
    if abriv == 'efb' or abriv == 'ac' or abriv == 'pgw':
        colors = ['darkmagenta', 'm', 'hotpink']
    if abriv == 'mcb' or abriv == 'cartland' or abriv == 'mag' or abriv == 'darcy' or abriv == 'mi' or abriv == 'nl' or abriv == 'as':
        colors = ['firebrick', 'red', 'darkorange']
    if  abriv == 'cornwell' or abriv == 'csf' or abriv == 'po':
        colors = ['darkgreen', 'green', 'limegreen']
    if abriv == 'dg' or abriv == 'rh' or abriv == 'pg':
        colors = ['mediumblue', 'royalblue', 'dodgerblue']
    if abriv == 'dickens' or abriv == 'ad' or abriv == 'th' or abriv == 'ws':
        colors = ['dimgrey', 'darkgrey', 'lightgrey']

    # save figure with percentages
    plt.figure()
    patches, texts, autotexts = plt.pie(sizes, startangle=90, autopct='%.0f%%', colors=colors)
    plt.legend(patches, labels, loc="upper left")
    plt.axis('equal')
    plt.savefig(os.path.join(save_path + str(name+'_pie_percentages_only_ppl_read.png')))

    # save figure without percentages
    labels = [str('Enjoyed ' + str(int(round(enjoyed))) + '%'), str('Thought ok ' + str(int(round(okay))) + '%'), str('Not Enjoyed ' + str(int(round(not_liked))) + '%')]
    plt.figure()
    patches, texts = plt.pie(sizes, startangle=90, colors=colors)
    plt.legend(patches, labels, loc="upper left")
    plt.axis('equal')
    plt.savefig(os.path.join(save_path + str(name + '_pie_only_ppl_read.png')))
    
    # print percentage of readers who have read the novel
    percent_read = ((float(len(df.loc[df[abriv] == 4])) + len(df.loc[df[abriv] == 3]) + len(df.loc[df[abriv] == 2])) / master_df_len) * 100.0
    print 'Percentage of readers who have read ', name, ' is: ', round(percent_read, 2)

# read in csv
df = pd.read_csv(file_path)

names = ['Jane Austen', 'E.F. Benson', 'The Brontes', 'Fanny Burney', 'Margaret Cambell Barnes', 'Barbara Cartland', 'Agatha Christie', 'Bernard Cornwell', 'Diana Gabaldon', 'Mary Ann Gibbs', 'Clare Darcy', 'Charles Dickens', 'Alexandre Dumas', 'C.S. Forester', 'Elizabeth Gaskell', 'Philippa Gregory', 'Thomas Hardy', 'Robert Harris', 'Margaret Irwin', 'Norah Lofts', "Patrick O'Brian", 'Ann Radcliffe', 'Anya Seton', 'William Shakespeare', 'P.G. Wodehouse']
abbreviations = ['ja', 'efb', 'b', 'fb', 'mcb', 'cartland', 'ac', 'cornwell', 'dg', 'mag', 'darcy', 'dickens', 'ad', 'csf', 'eg', 'pg', 'th', 'rh', 'mi', 'nl', 'po', 'ar', 'as', 'ws', 'pgw']

# loop through each novelist
for num in range(0, len(names)):
    create_pie_chart(names[num], df, abbreviations[num], len(df))

print ('Finished!')
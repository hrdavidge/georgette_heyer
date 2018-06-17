import argparse

parser = argparse.ArgumentParser("Code to check text for non-ascii code")
parser.add_argument('file',nargs='?',default=argparse.SUPPRESS)
parser.add_argument('--file',dest='file',default=None)

args = parser.parse_args()
file_path = args.file

if file_path == None:
    quit("User must specify the path to the .txt file")

file = open(file_path, 'r')
text = file.read().strip().replace('\r', ' ').replace('\n', ' ').replace('\xc3\xa9', 'e').replace("\xe2\x80\x9d", '"')
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
# split the text file up into individual strings
text = text.split(' ')

# try looping through all the strings
for num in range(0, len(text)):
    print(num)
    print (text[num])
    text[num].decode('ascii')

print('No new non-ascii characters in the text!')
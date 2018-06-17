import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

# for colours: https://matplotlib.org/2.0.0/examples/color/named_colors.html

parser = argparse.ArgumentParser("Code to create frequency distribution, pie chart and save percentages showing readers' jobs")
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

# printing numbers of specific jobs
print 'The number of teachers are: ', len(df[df['Teacher'].notnull()])
print 'The number of librarians are: ', len(df[df['Librarian'].notnull()])
print 'The number of writers are: ', len(df[df['Writer'].notnull()])
print 'The number of people working in history are: ', len(df[df['History'].notnull()])
print 'The number of people working in the book industry are: ', len(df[df['books'].notnull()])
print 'The number of people who gave their jobs are : ', len(df['num'].unique())

# find job type frequency, then print top ten as a frequency diagram
new_df = df[df['type_name'].notnull()]
unique_jobs, jobs_count = np.unique(new_df['type_name'], return_counts=True)

# the top 12 - would have had top ten, but 10 = 11 = 12
unique_df = pd.DataFrame(data = {'unique_jobs': unique_jobs, 'jobs_count': jobs_count}, columns = ['unique_jobs', 'jobs_count'])

sorted_unique_df = unique_df.sort_values('jobs_count', ascending=False)

short_sorted_unique_jobs = sorted_unique_df['unique_jobs'].head(12)

# loop through new_df and just select rows which match short_sorted_unique_jobs
top_12_job_types = pd.DataFrame(columns = ['num', 'job', 'type_num', 'type_name', 'Teacher', 'Librarian', 'Writer', 'History', 'books'])
for num in range(0, len(short_sorted_unique_jobs)):
    temp_df = new_df.loc[new_df['type_name'] == short_sorted_unique_jobs.iloc[num]]
    frames = [top_12_job_types, temp_df]
    top_12_job_types = pd.concat(frames, ignore_index=True)

# create the frequency diagram
plt.figure()
top_12_job_types['type_name'].value_counts().plot(kind='bar', title = "The Top Twelve Job Types of Heyer's Readers", y = 'Frequency')
plt.tight_layout()
plt.savefig(os.path.join(save_path + 'top_12_jobs_freq.png'))

print('Finished!')
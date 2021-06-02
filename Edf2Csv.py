import os
import glob
import pandas as pd
import mne
import numpy as np

os.chdir('Data')
file_extension = '.csv'

all_filenames = [i for i in glob.glob(f"*{file_extension}")]


def convert():
    for i in all_filenames:
        data = mne.io.read_raw_edf(i, preload=False)
        header = ','.join(data.ch_names)
        np.savetxt(i.replace('.edf', '') + '.csv', data.get_data().T, delimiter=',', header=header)

def combine():
    combined_csv_data = pd.concat([pd.read_csv(f, delimiter=',') for f in all_filenames[:3]], ignore_index=True)
    print(type(combined_csv_data))
    print(combined_csv_data.shape)
    combined_csv_data.to_csv('datas.csv')

combine()
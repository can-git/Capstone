import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 15)


def plot():
    file_name = 's004e01'
    file = f'Dataset/{file_name}/'

    dict = {}
    for path in os.listdir(file):
        d = loadmat(file + path, squeeze_me=True)
        # print(d['hgS_070000'].item()[3][0][2].item()[17].max())

        x_data = np.append(d['hgS_070000'].item()[3][0][3][0][2].item()[8], d['hgS_070000'].item()[3][0][3][1][2].item()[8])
        x_data = np.append(x_data, d['hgS_070000'].item()[3][0][3][2][2].item()[8])

        y_data = np.append(d['hgS_070000'].item()[3][0][3][0][2].item()[9], d['hgS_070000'].item()[3][0][3][1][2].item()[9])
        y_data = np.append(y_data, d['hgS_070000'].item()[3][0][3][2][2].item()[9])

        dict[path.split('.')[1]] = y_data

    sample_y1 = [0 for i in range(768)]
    sample_y2 = [1 for i in range(768)]
    sample_y3 = [0 for i in range(768)]
    y = np.hstack((sample_y1, sample_y2, sample_y3)).ravel()

    data = pd.DataFrame(dict)
    data['Status'] = y

    # print(data)
    # plt.plot(data.iloc[:, :14], label=data.iloc[:, :14].columns)
    # plt.legend()
    # plt.show()
    data.to_csv(f'Dataset/{file_name}/{file_name}.csv')


plot()


from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CenterScreen import center_screen_geometry
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import LineCollection
import mpl_toolkits.axisartist as axisartist
import time
from threading import Thread
import seaborn as sns
import statistics


def analyse():
    file = 'Emotiv 30s EDF/S001/S001E01.edf'
    raw = mne.io.read_raw_edf(file, preload=True, verbose=0)
    # Apply a bandpass filter between 0.5 - 45 Hz
    raw.filter(0.5, 45)

    # Extract the data and convert from V to uV
    data = raw._data * 1e6
    sf = raw.info['sfreq']
    chan = raw.ch_names

    # Let's have a look at the data
    print('Chan =', chan)
    print('Sampling frequency =', sf, 'Hz')
    print('Data shape =', data.shape)


def plot():
    file = 'Emotiv 30s EDF/S001/S001E01.edf'
    data = mne.io.read_raw_edf(file, preload=True)

    edf_data, times = data.crop(0, 1)[:, :100]

    x = times
    y = edf_data[:, :]
    print(y.shape)

    annotation = [1, 0] * 50
    c = ['r' if a else 'g' for a in annotation]

    fig = plt.figure(figsize=(9, 8), dpi=101)
    plt.grid(False)
    plt.xticks(x, rotation=90)
    plt.xlabel("Time(ss)")

    plt.ylabel("Channels")
    fig.tight_layout()
    plt.margins(0.01, tight=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

    # df = pd.DataFrame({"Times": x, "Att": c, "Values": y })

    for i in y:
        plt.scatter(x, i, c=c, s=10)
        plt.plot(x, i, '#95a5a6')
    # print("SS = ", statistics.stdev(y))
    # print("Max = ", np.max(y))

    plt.show()


def plot2():
    file = 'Emotiv 30s EDF/S001/S001E01.edf'
    data = mne.io.read_raw_edf(file, preload=True)

    hz = 1 / (data.times[1] - data.times[0])
    # If you wish to get specific channels and time:
    edf_data, times = data[0, int(0 * hz): int(1 * hz)]

    # edf_data, times = data.crop(0,1)[8:9, :]
    # print(edf_data.shape)
    x = times
    y = edf_data[0]
    annotation = [1, 0] * 128

    c = ['r' if a else 'g' for a in annotation]
    liste = zip(x[:-1], y[:-1], x[1:], y[1:])
    lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in liste]
    print()
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
    print(lines)
    plt.plot(colored_lines)  # Set line color and width
    plt.show()


def plot3():
    # construct some data
    n = 10
    x = np.arange(n + 1)  # resampledTime
    y = np.random.randn(n + 1)  # modulusOfZeroNormalized
    annotation = [1, 0] * 5

    # set up colors
    c = ['r' if a else 'g' for a in annotation]
    liste = zip(x[:-1], y[:-1], x[1:], y[1:])

    # convert time series to line segments
    lines = [((x0, y0), (x1, y1)) for x0, y0, x1, y1 in liste]
    print(lines)
    colored_lines = LineCollection(lines, colors=c, linewidths=(2,))

    # plot data
    fig, ax = plt.subplots(1)
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    plt.show()


def plot4():
    file = 'Emotiv 30s EDF/S001/S001E01.edf'
    data = mne.io.read_raw_edf(file, preload=True)
    df = data.to_data_frame()
    print(df)


plot()


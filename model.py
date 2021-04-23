import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class model:
    def add_data(self, data, ratioTest=0.1):
        edf_data, times = data[:, :]
        X = np.array(edf_data[:, :-1])
        X = np.transpose(X)
        y = np.array([1, 0] * int(X.shape[0]/2)).reshape((-1, 1))
        print(X.shape)
        print(y.shape)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=ratioTest)

    def set_epoch(self, x):
        self._epoch = x

    def set_batch(self, x):
        self._batch = x

    def DecisionTreefit(self):
        self.dtree = DecisionTreeClassifier(random_state=0)
        self.dtree.fit(self.X_train, self.y_train)
        self.ypred = self.dtree.predict(self.X_test)
        return True

    def score(self):
        self.accdtree = accuracy_score(self.y_test, self.ypred)
        self.predtree = precision_score(self.y_test, self.ypred, average='macro')
        self.recalldtree = recall_score(self.y_test, self.ypred, average='macro')
        self.f1dtree = f1_score(self.y_test, self.ypred, average='macro')
        resultDict = {
            'Accuracy': round(self.accdtree, 4),
            'Precision': round(self.predtree, 4),
            'Recall': round(self.recalldtree, 4),
            'F1': round(self.f1dtree, 4)
        }
        return resultDict

    def train(self):
        file = 'Emotiv 30s EDF/S001/S001E01.edf'
        data = mne.io.read_raw_edf(file, preload=True)

        edf_data, times = data[:, :]

        X = np.array(edf_data[:, :])
        X = np.transpose(X)
        y = np.array([1, 0] * 23040).reshape((-1, 1))
        print(X.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))



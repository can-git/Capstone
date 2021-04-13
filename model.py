import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class model:
    def add_data(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()

    def set_epoch(self, x):
        self._epoch = x

    def set_batch(self, x):
        self._batch = x

    def reShape(self):
        self.train_X = self.train_X.reshape((-1, 28 * 28))
        self.test_X = self.test_X.reshape((-1, 28 * 28))

    def DecisionTreefit(self):
        self.dtree = DecisionTreeClassifier()
        self.dtree.fit(self.train_X, self.train_y)
        self.ypred = self.dtree.predict(self.test_X)
        return True

    def score(self):
        self.accdtree = accuracy_score(self.test_y, self.ypred)
        self.predtree = precision_score(self.test_y, self.ypred, average='macro')
        self.recalldtree = recall_score(self.test_y, self.ypred, average='macro')
        self.f1dtree = f1_score(self.test_y, self.ypred, average='macro')
        resultDict = {
            'Accuracy': round(self.accdtree, 4),
            'Precision': round(self.predtree, 4),
            'Recall': round(self.recalldtree, 4),
            'F1': round(self.f1dtree, 4)
        }
        return resultDict

    def pr(self):
        print(self.train_X.shape)

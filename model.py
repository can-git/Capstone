import tensorflow as tf
import matplotlib.pyplot as plt


class model:
    def __init__(self, epoch=1, batch=1):
        self._epoch = epoch
        self._batch = batch

    def set_epoch(self, x):
        self._epoch = x

    def set_batch(self, x):
        self._batch = x

    def pr(self):
        print("Epoch = ", self._epoch, " Batch = ", self._batch)

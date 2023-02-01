import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class model:
    modelItself = None

    def add_data(self, data, ratioTest=0.1):
        X = data.iloc[:, :14]
        y = data.iloc[:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=ratioTest)

    def DecisionTreeFit(self):
        self.dtree = DecisionTreeClassifier(random_state=0)
        self.dtree.fit(self.X_train, self.y_train)
        self.ypred = self.dtree.predict(self.X_test)
        return self.dtree

    def NaiveBayesFit(self):
        self.nbayes = GaussianNB()
        self.nbayes.fit(self.X_train, np.ravel(self.y_train))
        self.ypred = self.nbayes.predict(self.X_test)
        return self.nbayes

    def RandomForestFit(self):
        self.rforest = RandomForestClassifier()
        self.rforest.fit(self.X_train, np.ravel(self.y_train))
        self.ypred = self.rforest.predict(self.X_test)
        return self.rforest

    def SVCFit(self):
        self.svc = SVC()
        self.svc.fit(self.X_train, np.ravel(self.y_train))
        self.ypred = self.svc.predict(self.X_test)
        return self.svc

    def LogisticRegressionFit(self):
        self.lr = LogisticRegression()
        self.lr.fit(self.X_train, np.ravel(self.y_train))
        self.ypred = self.lr.predict(self.X_test)
        return self.lr

    def score(self, classifier_name, _modelItself):

        self.acc = accuracy_score(self.y_test, self.ypred)
        self.pred = precision_score(self.y_test, self.ypred, average='macro', labels=np.unique(self.ypred))
        self.recall = recall_score(self.y_test, self.ypred, average='macro')
        self.f1d = f1_score(self.y_test, self.ypred, average='macro')
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.y_test, self.ypred).ravel()
        resultDict = {
            'Accuracy': round(self.acc, 5),
            'Precision': round(self.pred, 5),
            'Recall': round(self.recall, 5),
            'F1': round(self.f1d, 5),
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn,
            'TN': self.tn
        }
        now = datetime.now()
        fixedNow = now.strftime('%d-%m-%y-%H-%M-%S')
        filename4Model = classifier_name.replace(' ', '') + '-' + fixedNow + '.sav'
        filename4Text = classifier_name.replace(' ', '') + '-' + fixedNow + '.txt'

        pickle.dump(_modelItself, open('Results/' + filename4Model, 'wb'))

        modelLog = open('Results/' + filename4Text, 'w')
        details = (
                '\nModel Name = ' + classifier_name +
                '\n\nDate = ' + now.strftime('%B %d %Y') +
                '\nTime = ' + now.strftime('%H:%M:%S') +
                '\n\n---SCORES---\nAccuracy: ' + str(self.acc) +
                '\nPrecision: ' + str(self.pred) +
                '\nRecall: ' + str(self.recall) +
                '\nF1: ' + str(self.f1d) +
                '\n\n---Confusion Matrix---\nTP: ' + str(self.tp) +
                '\nFP: ' + str(self.fp) +
                '\nFN: ' + str(self.fn) +
                '\nTN: ' + str(self.tn)
        )
        modelLog.writelines(details)
        modelLog.close()
        return resultDict

    def show_test_results(self, df):
        signals = self.X_test
        labels = self.ypred
        x = np.array(signals.index)

        c = ['g' if a else 'r' for a in labels]

        fig = plt.figure(figsize=(16, 8), dpi=103)

        plt.grid(True)

        plt.xlabel("Time Points")
        plt.ylabel("uV")
        fig.tight_layout()
        plt.margins(0.01, tight=True)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

        for (icol, ival) in signals.iteritems():
            plt.scatter(x, ival.values, c=c, s=10)
        # plt.plot(df, '#95a5a6')
        plt.show()

    def predict(self, model, data):
        # edf_data, times = data[:, :]
        data = data.drop(data.columns[0], axis=1)
        x = data.iloc[:, :14]
        y = data.iloc[:, -1]

        resultPredict = model.predict(x)
        return 'Prediction is Successfull', resultPredict, x

    def show_prediction_results(self, labels, signals, low=0, high=100):
        signals = signals.iloc[low:high, :]
        labels = labels[low:high]
        x = np.array(signals.index)

        c = ['g' if a else 'r' for a in labels]

        fig = plt.figure(figsize=(16, 8), dpi=103)

        plt.grid(True)

        plt.xlabel("Time Points")
        plt.ylabel("uV")
        fig.tight_layout()
        plt.margins(0.01, tight=True)

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)

        for (icol,ival) in signals.iteritems():
            plt.scatter(x, ival.values, c=c, s=10)
        plt.plot(signals, '#95a5a6')
        plt.savefig("Results/{}.png".format("result"))
        plt.show()

import pandas as pd
import numpy as np
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import mne
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CenterScreen import center_screen_geometry
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import LineCollection
import mpl_toolkits.axisartist as axisartist
import model as m
import time
from threading import Thread
import pickle


class MyApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.root.title("Detection Interface")
        self.root.resizable(False, False)

        self.defaultBackground()
        # self.displayF()
        self.settingsF()
        self.ModelEvaluationF()
        self.DetectionF()

    def defaultBackground(self):
        # region MENU
        self.menu = tk.Menu(root)
        self.fileMenu = tk.Menu(self.menu, tearoff=0)
        self.fileMenu.add_command(label="Open EDF", command=self.open_file)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit")
        self.menu.add_cascade(label="File", menu=self.fileMenu)
        self.helpMenu = tk.Menu(self.menu, tearoff=0)
        self.helpMenu.add_command(label="Help")
        self.menu.add_cascade(label="Help", menu=self.helpMenu)
        self.root.config(menu=self.menu)
        # endregion

        # region GRID
        self.frameLeft = tk.LabelFrame(self)
        self.frameLeft.pack(side=tk.LEFT, fill='both')
        self.frameRight = tk.LabelFrame(self)
        self.frameRight.pack(side=tk.RIGHT, fill='both')
        # endregion

        # region NOTEBOOK
        self.notebook = ttk.Notebook(self.frameLeft)
        self.notebook.pack(fill='both')
        self.tab_Display = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Settings = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Model = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Detection = tk.Frame(self.notebook, width=200, height=700)
        self.notebook.add(self.tab_Display, text="Display")
        self.notebook.add(self.tab_Settings, text="Settings")
        self.notebook.add(self.tab_Model, text="Model Evaluation")
        self.notebook.add(self.tab_Detection, text="Detection")
        # endregion

    # region Visualize Right Side
    def plot(self, data):
        x = data.iloc[:, :14]
        y = data.iloc[:, -1]

        fig = plt.figure(figsize=(9, 8), dpi=99)
        fig.clf()
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.grid(True)
        ax.axis["left"].set_axisline_style("->", size=1.5)  # Set the y-axis style to ->arrow
        ax.axis["bottom"].set_axisline_style("->", size=1.5)  # Set the x-axis style to ->arrow
        ax.axis["top"].set_visible(False)  # Hide the above axis
        ax.axis["right"].set_visible(False)  # Hide the right axis

        plt.xlabel("Time Points")
        plt.ylabel("uV")

        plt.plot(x, label=x.columns, linewidth=0.5)  # Set line color and width
        plt.legend()
        canvas = FigureCanvasTkAgg(fig, master=self.frameRight)
        canvas.get_tk_widget().grid(row=0, column=0)
        canvas.draw()
    # endregion

    # region DisplayF
    def show_Channels(self, data):
        data_temp = data.copy()
        for i in self.ckbxVal:
            if self.ckbxVal.get(i).get() == 0:
                data_temp = data_temp.drop_channels([i])
        self.plot(data_temp)

    def open_file(self):
        self.file = filedialog.askopenfilename()
        self.data = pd.read_csv(self.file)
        self.data = self.data.drop(self.data.columns[0], axis=1)
        data_copy = self.data.copy()
        self.plot(data_copy)
        self.displayF(self.data)
        return self.data

    def displayF(self, data):
        self.ckbxVal = dict()
        num = 0
        for i in data.iloc[:, :14].columns:
            lblChannel = tk.Label(self.tab_Display, text=f'Channel {num+1}: {i}')
            lblChannel.grid(row=num, column=0, padx=10, pady=10, sticky=tk.W)
            num = num + 1
    # endregion

    # region SettingsF
    def settingsF(self):
        # region Data
        data_frame = tk.LabelFrame(self.tab_Settings, text="Data Split and Classifiers")
        data_frame.grid(row=0, column=0, padx=10, pady=10)

        lblTestD = tk.Label(data_frame, text="Test Data")
        lblTestD.grid(row=1, column=0, padx=10, pady=10)
        lblRatio2 = tk.Label(data_frame, text="Ratio")
        lblRatio2.grid(row=1, column=1, padx=10, pady=10)
        self.eTest = tk.DoubleVar(value=0.2)
        entryTest = tk.Entry(data_frame, textvariable=self.eTest, width=10)
        entryTest.grid(row=1, column=2, padx=10, pady=10)

        lblClassifier = tk.Label(data_frame, text="Classifier")
        lblClassifier.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

        dictClassifiers = tk.StringVar()
        self.comboClassifiers = ttk.Combobox(data_frame, width=25, textvariable=dictClassifiers, state="readonly")
        self.comboClassifiers.grid(row=2, column=1, columnspan=2, padx=10, pady=10)
        self.comboClassifiers['values'] = ('Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Support Vector Machine')
    # endregion

    # region ModelEvaluationF
    def ModelEvaluationF(self):
        # region Functions
        def result_scores(dict):
            self.eAcc.set(dict.get('Accuracy'))
            self.ePrecision.set(dict.get('Precision'))
            self.eRecall.set(dict.get('Recall'))
            self.eF1.set(dict.get('F1'))
            self.eTP.set(dict.get('TP'))
            self.eFP.set(dict.get('FP'))
            self.eFN.set(dict.get('FN'))
            self.eTN.set(dict.get('TN'))

        def progress_bar(x, text):
            self.lblProgress['text'] = text
            progress['value'] = x
            root.update_idletasks()
            time.sleep(0.5)

        def model_training():
            self.lbl_visualize_ready_test.config(text='Datas are not ready to visualize')
            progress_bar(10, 'Getting Data ..')
            
            progress_bar(20, 'Getting Data ...')
            
            progress_bar(30, 'Adjusting Data Ratio')
            m.model.add_data(self, self.data, self.eTest.get())

            modelItself = None
            progress_bar(50, 'Creating Model')
            if self.comboClassifiers.get() == "Decision Tree":
                modelItself = m.model.DecisionTreeFit(self)
            elif self.comboClassifiers.get() == "Naive Bayes":
                modelItself = m.model.NaiveBayesFit(self)
            elif self.comboClassifiers.get() == "Random Forest":
                modelItself = m.model.RandomForestFit(self)
            elif self.comboClassifiers.get() == "Support Vector Machine":
                modelItself = m.model.SVCFit(self)
            elif self.comboClassifiers.get() == "Logistic Regression":
                modelItself = m.model.LogisticRegressionFit(self)
            
            progress_bar(90, 'Results Coming')
            result_scores(m.model.score(self, self.comboClassifiers.get(), modelItself))
            
            progress_bar(100, 'Completed..')

            self.lbl_visualize_ready_test.config(text='Datas are ready to visualize')

        def btn_trigger():
            Thread(target=model_training).start()
            Thread(target=progress_bar(0, '')).start()

        def btn_visualize_test():
            m.model.show_test_results(self, self.data)

        createB = tk.Button(self.tab_Model, text="Create and Test", command=btn_trigger)
        createB.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        progress = ttk.Progressbar(self.tab_Model, orient=tk.HORIZONTAL, length=120, mode='determinate')
        progress.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        self.lblProgress = tk.Label(self.tab_Model, text='')
        self.lblProgress.grid(row=1, column=1, padx=2, pady=2, sticky=tk.E)
        # endregion

        # region Evaluation
        performance_frame = tk.LabelFrame(self.tab_Model, text="Performance")
        performance_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        lblAcc = tk.Label(performance_frame, text="Accuracy")
        lblAcc.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        lblAcc.config(width=18)
        lblPrecision = tk.Label(performance_frame, text="Precision")
        lblPrecision.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lblPrecision.config(width=18)
        lblRecall = tk.Label(performance_frame, text="Recall")
        lblRecall.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        lblRecall.config(width=18)
        lblF1 = tk.Label(performance_frame, text="F1")
        lblF1.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        lblF1.config(width=18)

        self.eAcc = tk.DoubleVar(value=0.0)
        entryAcc = tk.Entry(performance_frame, textvariable=self.eAcc, width=15, state='readonly', justify=tk.CENTER)
        entryAcc.grid(row=0, column=1, padx=10, pady=10, sticky=tk.E)
        self.ePrecision = tk.DoubleVar(value=0.0)
        entryPrecision = tk.Entry(performance_frame, textvariable=self.ePrecision, width=15, state='readonly', justify=tk.CENTER)
        entryPrecision.grid(row=1, column=1, padx=10, pady=10, sticky=tk.E)
        self.eRecall = tk.DoubleVar(value=0.0)
        entryRecall = tk.Entry(performance_frame, textvariable=self.eRecall, width=15, state='readonly', justify=tk.CENTER)
        entryRecall.grid(row=2, column=1, padx=10, pady=10, sticky=tk.E)
        self.eF1 = tk.DoubleVar(value=0.0)
        entryF1 = tk.Entry(performance_frame, textvariable=self.eF1, width=15, state='readonly', justify=tk.CENTER)
        entryF1.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)

        # region Confusion Matrix
        matrix_frame = tk.LabelFrame(self.tab_Model, text="Confusion Matrix")
        matrix_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        lblTP = tk.Label(matrix_frame, text="True Positives")
        lblTP.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        lblTP.config(width=18)
        lblFP = tk.Label(matrix_frame, text="False Positives")
        lblFP.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lblFP.config(width=18)
        lblFN = tk.Label(matrix_frame, text="False Negatives")
        lblFN.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        lblFN.config(width=18)
        lblTN = tk.Label(matrix_frame, text="True Negatives")
        lblTN.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        lblTN.config(width=18)

        self.eTP = tk.DoubleVar(value=0.0)
        entryTP = tk.Entry(matrix_frame, textvariable=self.eTP, width=15, state='readonly', justify=tk.CENTER)
        entryTP.grid(row=0, column=1, padx=10, pady=10, sticky=tk.E)
        self.eFP = tk.DoubleVar(value=0.0)
        entryFP = tk.Entry(matrix_frame, textvariable=self.eFP, width=15, state='readonly', justify=tk.CENTER)
        entryFP.grid(row=1, column=1, padx=10, pady=10, sticky=tk.E)
        self.eFN = tk.DoubleVar(value=0.0)
        entryFN = tk.Entry(matrix_frame, textvariable=self.eFN, width=15, state='readonly', justify=tk.CENTER)
        entryFN.grid(row=2, column=1, padx=10, pady=10, sticky=tk.E)
        self.eTN = tk.DoubleVar(value=0.0)
        entryTN = tk.Entry(matrix_frame, textvariable=self.eTN, width=15, state='readonly', justify=tk.CENTER)
        entryTN.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)
        # endregion

        # region Visualize
        test_frame = tk.LabelFrame(self.tab_Model, text="Test")
        test_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

        visualize_testB = tk.Button(test_frame, text="Visualize", width=25, command=btn_visualize_test)
        visualize_testB.grid(row=0, column=0, padx=10, pady=10)

        self.lbl_visualize_ready_test = tk.Label(test_frame, text="Datas are not ready to visualize")
        self.lbl_visualize_ready_test.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        # endregion
    # endregion

    # region DetectionF
    def DetectionF(self):
        self.model_file = None
        self.data_file = None
        self.labels = None
        self.original_x = None
        self.cache = list()
        def btn_load_model():
            self.model_file_name = filedialog.askopenfilename()
            self.model_file = pickle.load(open(self.model_file_name,'rb'))
            self.lbl_model_name.config(text = 'Model Loaded')

        def btn_load_data():
            self.data_file_name = filedialog.askopenfilename()
            self.data_file = pd.read_csv(self.data_file_name)
            self.lbl_data_name.config(text = 'Data Loaded')

        def btn_predict_data():
            isSuccess, self.labels, self.original_x  = m.model.predict(self, self.model_file, self.data_file)
            if isSuccess:
                self.lbl_predict_ready.config(text='Datas are predicted successfully')
                self.lbl_visualize_ready.config(text='Datas are ready to visualize')

                for i in range(len(self.labels)):
                    self.cache.append(i)
                self.comboMin['values'] = self.cache
                self.comboMax['values'] = self.cache
            else:
                self.lbl_predict_ready.config(text='Datas are not predicted')
                self.lbl_visualize_ready.config(text='Datas are not ready to visualize')

        def btn_visualize_data():
            m.model.show_prediction_results(self, self.labels, self.original_x, int(self.comboMin.get()), int(self.comboMax.get()))

        detection_frame = tk.LabelFrame(self.tab_Detection, text="Detection")
        detection_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky=tk.N)

        load_modelB = tk.Button(detection_frame, text="Load Model", width=35, command=btn_load_model)
        load_modelB.grid(row=0, column=0, padx=10, pady=10)

        self.lbl_model_name = tk.Label(detection_frame, text="Model is not selected")
        self.lbl_model_name.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        load_DataB = tk.Button(detection_frame, text="Load Data", width=35, command=btn_load_data)
        load_DataB.grid(row=2, column=0, padx=10, pady=10)

        self.lbl_data_name = tk.Label(detection_frame, text="Data is not selected")
        self.lbl_data_name.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

        predict_modelB = tk.Button(detection_frame, text="Predict Data", width=35, command=btn_predict_data)
        predict_modelB.grid(row=4, column=0, padx=10, pady=10)

        self.lbl_predict_ready = tk.Label(detection_frame, text="Datas are not predicted")
        self.lbl_predict_ready.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

        self.lbl_min = tk.Label(detection_frame, text="Min Time")
        self.lbl_min.grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)

        min = tk.IntVar()
        self.comboMin = ttk.Combobox(detection_frame, width=8, textvariable=min)
        self.comboMin.grid(row=7, column=0, padx=5, pady=5)
        self.comboMin['values'] = self.cache

        self.lbl_max = tk.Label(detection_frame, text="Max Time")
        self.lbl_max.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)

        max = tk.IntVar()
        self.comboMax = ttk.Combobox(detection_frame, width=8, textvariable=max)
        self.comboMax.grid(row=9, column=0, padx=5, pady=5)
        self.comboMax['values'] = self.cache

        visualize_modelB = tk.Button(detection_frame, text="Visualize", width=35, command=btn_visualize_data)
        visualize_modelB.grid(row=10, column=0, padx=10, pady=10)

        self.lbl_visualize_ready = tk.Label(detection_frame, text="Datas are not ready to visualize")
        self.lbl_visualize_ready.grid(row=11, column=0, padx=5, pady=5, sticky=tk.W)

    # endregion


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(center_screen_geometry(screen_width=root.winfo_screenwidth(),
                                         screen_height=root.winfo_screenheight(),
                                         window_width=1200,
                                         window_height=800))
    MyApp(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CenterScreen import center_screen_geometry
from matplotlib.pyplot import MultipleLocator
from matplotlib.collections import LineCollection
import mpl_toolkits.axisartist as axisartist
import model as m
import time
from threading import Thread


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
    def plot(self):
        file = 'Emotiv 30s EDF/S001/S001E01.edf'
        data = mne.io.read_raw_edf(file, preload=True)
        tmin = 0
        tmax = 10
        chans = data.ch_names
        selection = data.crop(tmin, tmax)
        selection = selection.pick_channels(chans)

        sl = selection[:, :]  # extract into array format

        offset = np.arange(0, 14 * 0.001, 0.001)
        x = sl[1]  # x axis data
        y = sl[0].T + offset  # y-axis data

        ylabel = chans  # y-axis tick name

        fig = plt.figure(figsize=(9, 8), dpi=101)
        ax = axisartist.Subplot(fig, 111)

        fig.add_axes(ax)
        ax.axis["left"].set_axisline_style("->", size=1.5)  # Set the y-axis style to ->arrow
        ax.axis["bottom"].set_axisline_style("->", size=1.5)  # Set the x-axis style to ->arrow
        ax.axis["top"].set_visible(False)  # Hide the above axis
        ax.axis["right"].set_visible(False)  # Hide the right axis
        x_major_locator = MultipleLocator(10)  # Set the scale spacing to 1
        ax.xaxis.set_major_locator(x_major_locator)
        plt.yticks(offset.tolist(), ylabel)  # Modify the name of the y-axis scale
        plt.xlabel("Time(s)")
        plt.ylabel("Channels")
        fig.tight_layout()
        plt.margins(0.01, tight=True)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
        plt.plot(x, y, linewidth=0.5)  # Set line color and width

        canvas = FigureCanvasTkAgg(fig, master=self.frameRight)
        canvas.get_tk_widget().grid(row=0, column=0)

        canvas.draw()

    def plot2(self, data):
        fig = plt.figure(figsize=(9, 8), dpi=101)
        fig.clf()
        ax = axisartist.Subplot(fig, 111)
        fig.add_axes(ax)
        ax.grid(True)
        ax.axis["left"].set_axisline_style("->", size=1.5)  # Set the y-axis style to ->arrow
        ax.axis["bottom"].set_axisline_style("->", size=1.5)  # Set the x-axis style to ->arrow
        ax.axis["top"].set_visible(False)  # Hide the above axis
        ax.axis["right"].set_visible(False)  # Hide the right axis
        x_major_locator = MultipleLocator(1000)  # Set the scale spacing to 1
        ax.xaxis.set_major_locator(x_major_locator)
        plt.xlabel("Time(ss)")
        plt.ylabel("uV")
        fig.tight_layout()
        plt.margins(0.01, tight=True)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
        plt.plot(np.transpose(data.crop(0, 100).get_data()), label=data.ch_names, linewidth=0.5)  # Set line color and width
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
        self.plot2(data_temp)


    def open_file(self):
        self.file = filedialog.askopenfilename()
        self.data = mne.io.read_raw_edf(self.file, preload=True)
        self.plot2(self.data)
        self.displayF(self.data)
        #print(self.data.get_data()[0])
        #self.data.plot()
        return self.data

    def displayF(self, data):
        self.ckbxVal = dict()
        num = 0
        for i in data.ch_names:
            self.ckbxVal[i] = tk.IntVar(value=1)
            ckbx = tk.Checkbutton(self.tab_Display, text=i, variable=self.ckbxVal[i], command=lambda key=i: self.show_Channels(data))
            ckbx.grid(row=num, column=0, padx=10, pady=10, sticky=tk.W)

            label = tk.Label(self.tab_Display, text="Amp")
            label.grid(row=num, column=2, padx=10, pady=10)

            e = tk.DoubleVar(value=256.0)
            entry = tk.Entry(self.tab_Display, textvariable=e, width=10)
            entry.grid(row=num, column=3, padx=10, pady=10)

            num = num + 1
            if i == data.ch_names[-1]:
                label = tk.Label(self.tab_Display, text="Set default amplitude as")
                label.grid(row=num, column=0, padx=10, pady=10)

                e = tk.DoubleVar(value=256.0)
                entry= tk.Entry(self.tab_Display, textvariable=e, width=10)
                entry.grid(row=num, column=3, padx=10, pady=10)
    # endregion

    # region SettingsF
    def settingsF(self):
        # region Data
        data_frame = tk.LabelFrame(self.tab_Settings, text="Data")
        data_frame.grid(row=0, column=0, padx=10, pady=10)

        lblTrainD = tk.Label(data_frame, text="Train Data")
        lblTrainD.grid(row=0, column=0, padx=10, pady=10)
        lblTestD = tk.Label(data_frame, text="Test Data")
        lblTestD.grid(row=1, column=0, padx=10, pady=10)
        lblRatio1 = tk.Label(data_frame, text="Ratio")
        lblRatio1.grid(row=0, column=1, padx=10, pady=10)
        lblRatio2 = tk.Label(data_frame, text="Ratio")
        lblRatio2.grid(row=1, column=1, padx=10, pady=10)
        self.eTrain = tk.DoubleVar(value=0.8)
        entryTrain = tk.Entry(data_frame, textvariable=self.eTrain, width=10)
        entryTrain.grid(row=0, column=2, padx=10, pady=10)
        self.eTest = tk.DoubleVar(value=0.2)
        entryTest = tk.Entry(data_frame, textvariable=self.eTest, width=10)
        entryTest.grid(row=1, column=2, padx=10, pady=10)

        self.valNormalizing = tk.IntVar(value=1)
        ckbxNormalizing = tk.Checkbutton(data_frame, text="Normalizing", variable=self.valNormalizing)
        ckbxNormalizing.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.valMean = tk.IntVar()
        ckbxMean = tk.Checkbutton(data_frame, text="Mean", variable=self.valMean)
        ckbxMean.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
        # endregion

        # region Model
        model_frame = tk.LabelFrame(self.tab_Settings, text="Model")
        model_frame.grid(row=1, column=0, padx=10, pady=10)

        lblClassifier = tk.Label(model_frame, text="Classifier")
        lblClassifier.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        dictClassifiers = tk.StringVar()
        comboClassifiers = ttk.Combobox(model_frame, width=25, textvariable=dictClassifiers, state="readonly")
        comboClassifiers.grid(row=0, column=1, padx=10, pady=10)
        comboClassifiers['values'] = ('Naive Bayes', 'Decision Tree', 'Random Forest')

        lblEpoch = tk.Label(model_frame, text="Epochs")
        lblEpoch.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lblBacth = tk.Label(model_frame, text="Batch Size")
        lblBacth.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.eEpoch = tk.IntVar(value=1)
        entryEpoch = tk.Entry(model_frame, textvariable=self.eEpoch, width=10)
        entryEpoch.grid(row=1, column=1, padx=10, pady=10, sticky=tk.E)
        self.eBatch = tk.IntVar(value=1)
        entryBatch = tk.Entry(model_frame, textvariable=self.eBatch, width=10)
        entryBatch.grid(row=2, column=1, padx=10, pady=10, sticky=tk.E)
        # endregion

        # region Filter
        filter_frame = tk.LabelFrame(self.tab_Settings, text="Filtering")
        filter_frame.grid(row=2, column=0, padx=10, pady=10)

        valEnable = tk.IntVar(value=1)
        ckbxEnable = tk.Checkbutton(filter_frame, text="Enable", variable=valEnable)
        ckbxEnable.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        lblDisplay = tk.Label(filter_frame, text="Display")
        lblDisplay.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lblFilterT = tk.Label(filter_frame, text="Filter Type")
        lblFilterT.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

        dictDisplays = tk.StringVar()
        comboDisplay = ttk.Combobox(filter_frame, width=25, textvariable=dictDisplays, state="readonly")
        comboDisplay.grid(row=1, column=1, columnspan=4, padx=10, pady=10)
        comboDisplay['values'] = ('Filter')
        comboDisplay.current(0)
        dictFilters = tk.StringVar()
        comboFilter = ttk.Combobox(filter_frame, width=25, textvariable=dictFilters, state="readonly")
        comboFilter.grid(row=2, column=1, columnspan=4, padx=10, pady=10)
        comboFilter['values'] = ("Bandpass")
        comboFilter.set(0)

        lblFreq = tk.Label(filter_frame, text="Frequency")
        lblFreq.grid(row=3, column=0, pady=10)
        lblPmin = tk.Label(filter_frame, text="Pmin")
        lblPmin.grid(row=3, column=1)
        lblPmax = tk.Label(filter_frame, text="Pmax")
        lblPmax.grid(row=3, column=3)
        ePmin = tk.DoubleVar(value=12.0)
        entryPmin = tk.Entry(filter_frame, textvariable=ePmin, width=5)
        entryPmin.grid(row=3, column=2)
        ePmax = tk.DoubleVar(value=14.0)
        entryPmax = tk.Entry(filter_frame, textvariable=ePmax, width=5)
        entryPmax.grid(row=3, column=4)

        # endregion
    # endregion

    # region ModelEvaluationF
    def ModelEvaluationF(self):
        # region Create
        def result_scores(dict):
            self.eAcc.set(dict.get('Accuracy'))
            self.ePrecision.set(dict.get('Precision'))
            self.eRecall.set(dict.get('Recall'))
            self.eF1.set(dict.get('F1'))

        def progress_bar(x, text):
            self.lblProgress['text'] = text
            progress['value'] = x
            root.update_idletasks()
            time.sleep(0.5)

        def model_training():
            progress_bar(10, 'Getting Epoch Value')
            m.model.set_epoch(self, self.eEpoch.get())
            progress_bar(20, 'Getting Batch Size Value')
            m.model.set_batch(self, self.eBatch.get())
            progress_bar(30, 'Adding Train and Test Data')
            m.model.add_data(self, self.data, self.eTest.get())
            progress_bar(50, 'Creating Model')
            m.model.DecisionTreefit(self)
            progress_bar(90, 'Results Coming')
            result_scores(m.model.score(self))
            progress_bar(100, 'Completed..')

        def btn_trigger():
            Thread(target=model_training).start()
            Thread(target=progress_bar(0, '')).start()

        self.lblProgress = tk.Label(self.tab_Model, text='')
        self.lblProgress.grid(row=1, column=1, padx=2, pady=2, sticky=tk.E)
        createB = tk.Button(self.tab_Model, text="Create and Test", command=btn_trigger)
        createB.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        progress = ttk.Progressbar(self.tab_Model, orient=tk.HORIZONTAL, length=150, mode='determinate')
        progress.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)
        # endregion

        # region Evaluation
        performance_frame = tk.LabelFrame(self.tab_Model, text="Performance")
        performance_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

        lblAcc = tk.Label(performance_frame, text="Accuracy")
        lblAcc.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        lblLoss = tk.Label(performance_frame, text="Loss")
        lblLoss.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        lblPrecision = tk.Label(performance_frame, text="Precision")
        lblPrecision.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        lblRecall = tk.Label(performance_frame, text="Recall")
        lblRecall.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        lblF1 = tk.Label(performance_frame, text="F1")
        lblF1.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)

        self.eAcc = tk.DoubleVar(value=0.0)
        entryAcc = tk.Entry(performance_frame, textvariable=self.eAcc, width=15, state='readonly', justify=tk.CENTER)
        entryAcc.grid(row=0, column=1, padx=10, pady=10, sticky=tk.E)
        self.eLoss = tk.DoubleVar(value=0.0)
        entryLoss = tk.Entry(performance_frame, textvariable=self.eLoss, width=15, state='readonly', justify=tk.CENTER)
        entryLoss.grid(row=1, column=1, padx=10, pady=10, sticky=tk.E)
        self.ePrecision = tk.DoubleVar(value=0.0)
        entryPrecision = tk.Entry(performance_frame, textvariable=self.ePrecision, width=15, state='readonly', justify=tk.CENTER)
        entryPrecision.grid(row=2, column=1, padx=10, pady=10, sticky=tk.E)
        self.eRecall = tk.DoubleVar(value=0.0)
        entryRecall = tk.Entry(performance_frame, textvariable=self.eRecall, width=15, state='readonly', justify=tk.CENTER)
        entryRecall.grid(row=3, column=1, padx=10, pady=10, sticky=tk.E)
        self.eF1 = tk.DoubleVar(value=0.0)
        entryF1 = tk.Entry(performance_frame, textvariable=self.eF1, width=15, state='readonly', justify=tk.CENTER)
        entryF1.grid(row=4, column=1, padx=10, pady=10, sticky=tk.E)
        # endregion
    # endregion

    # region DetectionF
    def DetectionF(self):
        pass
    # endregion


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(center_screen_geometry(screen_width=root.winfo_screenwidth(),
                                         screen_height=root.winfo_screenheight(),
                                         window_width=1200,
                                         window_height=800))
    MyApp(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

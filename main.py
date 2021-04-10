from tkinter import ttk
import tkinter as tk
from CenterScreen import center_screen_geometry


class MyApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.root.title("LMS")
        self.root.resizable(False, False)

        self.defaultBackground()

        self.displayF()
        self.settingsF()

    def defaultBackground(self):
        # MENU
        self.menu = tk.Menu(root)
        self.fileMenu = tk.Menu(self.menu, tearoff=0)
        self.fileMenu.add_command(label="New Database")
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit")
        self.menu.add_cascade(label="File", menu=self.fileMenu)
        self.helpMenu = tk.Menu(self.menu, tearoff=0)
        self.helpMenu.add_command(label="Help")
        self.menu.add_cascade(label="Help", menu=self.helpMenu)
        self.root.config(menu=self.menu)

        # GRID
        self.frameLeft = tk.LabelFrame(self)
        self.frameLeft.grid(row=0, column=1, padx=10, pady=10)
        self.frameRight = tk.LabelFrame(self)
        self.frameRight.grid(row=0, column=2, padx=10, pady=10)

        self.lbl_frameRight = tk.Label(self.frameRight, text="frameRight")
        self.lbl_frameRight.grid(row=0, column=0, padx=5, pady=5)

        # NOTEBOOK
        self.notebook = ttk.Notebook(self.frameLeft)
        self.notebook.pack(fill='both', expand=True)
        self.tab_Display = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Settings = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Model = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Evaluation = tk.Frame(self.notebook, width=200, height=700)
        self.tab_Detection = tk.Frame(self.notebook, width=200, height=700)
        self.notebook.add(self.tab_Display, text="Display")
        self.notebook.add(self.tab_Settings, text="Settings")
        self.notebook.add(self.tab_Model, text="Model")
        self.notebook.add(self.tab_Evaluation, text="Evaluation")
        self.notebook.add(self.tab_Detection, text="Detection")

    def displayF(self):
        self.lbl = tk.Label(self.tab_Display, text="display")
        self.lbl.grid(row=0, column=0, padx=10, pady=10)

    def settingsF(self):
        self.lbl2 = tk.Label(self.tab_Settings, text="settings")
        self.lbl2.grid(row=0, column=0, padx=10, pady=10)


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry(center_screen_geometry(screen_width=root.winfo_screenwidth(),
                                         screen_height=root.winfo_screenheight(),
                                         window_width=1200,
                                         window_height=800))
    MyApp(root).pack(side="top", fill="both", expand=True)
    root.mainloop()

# %% COMBINE MASTER GAIT DATA SHEETS FROM EACH DAY TO FORM THE ULTIMATE DATA SHEET

# %% IMPORTS
import numpy as np
from tkinter import filedialog
import os
import pandas as pd

# %% DEFINE CLASS
class combine_master_sheets:
    def __init__(self, overall_gait_folder):
        # INITIALIZE CLASS VARIABLES
        self.overall_gait_folder = overall_gait_folder

        # RUN CLASS FUNCTIONS
        self.create_excel()
        self.find_paths()

    def create_excel(self):
        pass

    def find_paths(self):
        temp_folder_list = os.listdir(self.overall_gait_folder)
        self.folder_list = sorted([item for item in temp_folder_list if 'analyzed' in item])
        self.path_list = [os.path.join(self.overall_gait_folder,item,"MASTER_GAIT.xlsx") for item in self.folder_list]
        self.master_found = [os.path.isfile(item) for item in self.path_list]
        for folder,found in zip(self.folder_list,self.master_found): print(folder + ", master found: " + str(found))

# %% RUN COMBINE_MASTER_SHEETS IF THIS IS A STANDALONE RUN
if __name__ == '__main__':
    #overall_gait_folder = filedialog.askdirectory(title="Select Folder")
    overall_gait_folder = "/Users/paveyboys/Desktop/TOP_VELOCITY/GAIT"
    combine_master_sheets(overall_gait_folder)
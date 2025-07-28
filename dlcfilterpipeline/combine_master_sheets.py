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
        self.path_list = []
        self.dataframe_list = []
        self.combined_dataframe = None

        # RUN CLASS FUNCTIONS
        self.find_paths()
        self.aggregate_data()
        self.create_excel()

    def find_paths(self):
        temp_folder_list = os.listdir(self.overall_gait_folder)
        self.folder_list = sorted([item for item in temp_folder_list if 'analyzed' in item])
        self.path_list = [os.path.join(self.overall_gait_folder,item,item+"_MASTER.xlsx") for item in self.folder_list]
        self.master_found = [os.path.isfile(item) for item in self.path_list]
        for folder,found in zip(self.folder_list,self.master_found): print(folder + ", master found: " + str(found))

    def aggregate_data(self):
        for item in self.path_list:
            self.dataframe_list.append(pd.read_excel(item))
        self.combined_dataframe = pd.concat(self.dataframe_list)
        extracted = self.combined_dataframe['File'].str.extract(r'(?P<Date>[^_]+)_(?P<MouseID>[^_]+)_(?P<TestNumber>[^_]+)')
        self.combined_dataframe = pd.concat([self.combined_dataframe, extracted], axis=1)

    def create_excel(self):
        if self.combined_dataframe is not None:
            self.combined_dataframe.to_excel(os.path.join(self.overall_gait_folder,"MASTER_GAIT.xlsx"),index=False)

# %% RUN COMBINE_MASTER_SHEETS IF THIS IS A STANDALONE RUN
if __name__ == '__main__':
    overall_gait_folder = filedialog.askdirectory(title="Select Folder")
    combine_master_sheets(overall_gait_folder)
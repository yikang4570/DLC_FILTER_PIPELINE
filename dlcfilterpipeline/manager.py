#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A parent class, base plotter, which initializes plotters and holds common 'self'-referring functions
@author: Shawn Pavey
"""
# %% IMPORT PACKAGES
# Data packages
import numpy as np
import pandas as pd
import readyplot as rp
import tables

# Directory management
import os
import tkinter as tk
from tkinter import filedialog

# ML functions
from sklearn.cluster import DBSCAN

# Package utils
from .utils import (dict_update_nested,
                    mini_kwarg_resolver,
                    swap_ext)

# Package settings (error suppression)
pd.options.mode.chained_assignment = None

#%%---------------------------------------------------------------------------------------------------------------------
# CLASS MAIN
#-----------------------------------------------------------------------------------------------------------------------
# %% INITIALIZE CLASS
class Manager:
    def __init__(self,input_dict):
        self.input_dict = input_dict
        for name, value in input_dict.items(): setattr(self, name, value)

        if self.verbose:
            for name, value in self.__dict__.items(): print(f"{name}: {value}")

        if 'video_paths' not in input_dict: self.select_initial_files()

        if self.verbose:
            for name, value in self.__dict__.items(): print(f"{name}: {value}")

# %% MAIN METHODS
    def batch_process(self):
        self.fully_processed_list = []
        for path in self.video_paths:
            self.single_process(path)
            self.fully_processed_list.append(path)


    def single_process(self,path=None):
        # INITIALIZE VIDEO PATH
        if path is None: path = self.video_paths[0]
        elif isinstance(path, int): path = self.video_paths[path]
        self.current_path = path

        # ANALYZE VIDEO

        # MANAGE FILE NAMING CONVENTIONS TO FIND DATA
        if not hasattr(self, 'model_name'):
            self.model_name = self.get_text()
            print(self.model_name)
        else: print("model_name is already in the class as: ",self.model_name)
        self.data_file_name = swap_ext(self.current_path, self.model_name)

        # LOAD H5
        try:
            self.df = pd.read_hdf(self.data_file_name + '.h5')
            print("HDF5 file: " + self.data_file_name + '.h5' + " successfully loaded.")
        except Exception as e:
            print(f"Error reading HDF5 file:" + self.data_file_name + " ERROR: {e}")
            return

        # FILTER H5
        if self.verbose: print("Initial DataFrame:", self.df.head())
        self.df_processed = self.custom_filter(self.df, self.current_path)
        if self.verbose: print("Processed DataFrame:", self.df_processed.head())

        # SAVE FILES
        self.save() if self.save_files == True else print("Files not saving per user settings")


# %% INTERNAL METHODS
    def select_initial_files(self):
        root = tk.Tk()
        root.withdraw()
        self.video_paths = filedialog.askopenfilenames(
            title="Select Files", filetypes=[("Video Files", "*.*")],initialdir=self.defaultdir)
        for path in self.video_paths: print("Selected file: ", path)
        root.destroy()
        return

    def simplify_df(df):
        cs = ['frame', 'x', 'y', 'likelihood', 'bodyparts']
        scorer = df.columns.get_level_values(0).unique().tolist()[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()
        coords = df.columns.get_level_values(2).unique().tolist()
        df_simple = pd.DataFrame(columns=cs)

        for i in range(len(df)):
            tpdf = df.iloc[i]
            for bodypart in bodyparts:
                new_row = [i,
                           tpdf.loc[(scorer, bodypart, cs[1])],
                           tpdf.loc[(scorer, bodypart, cs[2])],
                           tpdf.loc[(scorer, bodypart, cs[3])],
                           bodypart]
                df_simple.loc[len(df_simple)] = new_row
        return df_simple

    def save(self):
        # SAVE H5
        try:
            self.df.to_hdf(self.data_file_name + '_backup.h5', key='df', mode='w')
            self.df_processed.to_hdf(self.data_file_name + '.h5', key='df', mode='w')
            print(f"Processed data successfully written for '{self.data_file_name}'.h5.")
        except Exception as e:
            print(f"Error writing: " + self.data_file_name + ".h5 to HDF5 file: {e}")
            return

        # SAVE CSV
        try:
            self.df.to_csv(self.data_file_name + '_backup.csv', index=True)
            self.df_processed.to_csv(self.data_file_name + '.csv', index=True)
            print(f"Processed data successfully saved as CSV to '{self.data_file_name}'.csv.")
        except Exception as e:
            print(f"Error writing:" + self.data_file_name + ".csv to CSV file: {e}")

    def get_text(self):
        def get_text_button():
            self.text = entry.get()
            root.withdraw()
            root.destroy()
        root = tk.Tk()
        root.title("Text Prompt Example")
        entry = tk.Entry(root, width=30)
        entry.pack(pady=10)
        button = tk.Button(root, text="Submit", command=get_text_button)
        button.pack(pady=5)
        root.mainloop()
        return self.text


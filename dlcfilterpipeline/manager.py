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
import subprocess
import json
from ruamel.yaml import YAML

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

        self.config_path = os.getcwd() + os.sep + "dlcfilterpipeline" + os.sep + "config.yaml"
        yaml = YAML()
        with open(self.config_path, 'r') as file: data = yaml.load(file)
        data['pcutoff'] = self.pcutoff
        with open(self.config_path, 'w') as file:
            yaml.dump(data, file)

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
        if self.analyze_videos:
            venv_python = "/opt/anaconda3/envs/DEEPLABCUT/bin/python"
            subprocess.run([
                venv_python,
                "dlcfilterpipeline/dlc_analyze.py",
                self.config_path,
                self.current_path,
                self.shuffle])

            with open("dlcfilterpipeline" + os.sep + "model_name.json", 'r') as openfile:
                json_object = json.load(openfile)
                self.model_name = json_object["model_name"]
                print("CREATED MODEL NAME IS: ",self.model_name)

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
            print(f"Error reading HDF5 file:" + self.data_file_name + f" ERROR: {e}")
            return

        # FILTER H5
        if self.verbose: print("Initial DataFrame:", self.df.head())
        self.df_processed = self.custom_filter()
        if self.verbose: print("Processed DataFrame:", self.df_processed.head())

        # SAVE FILES
        self.save() if self.save_files == True else print("Files not saving per user settings")

        # CREATE VIDEO
        if self.create_videos:
            venv_python = "/opt/anaconda3/envs/DEEPLABCUT/bin/python"
            subprocess.run([
                venv_python,
                "dlcfilterpipeline/dlc_create_video.py",
                self.config_path,
                self.current_path,
                self.shuffle,
                ",".join(self.bodyparts)])

        # ANIMATE VIDEO PLOT
        if self.animate: self.animator()

    def custom_filter(self):
        print(self.df.columns.names)
        self.processed_df = self.df
        self.plot_generator()

    def plot_generator(self,frame=None):
        title = self.current_path.split(os.sep)[-1]

        df_pre, df_post = self.simplify_df(self.df), self.simplify_df(self.processed_df)

        if frame is not None:
            combined_df = pd.concat([df_pre, df_post], axis=1)
            xlims = (combined_df['x'].min().min(), combined_df['x'].max().max())
            ylims = (combined_df['y'].min().min(), combined_df['y'].max().max())
            tlims = (combined_df['frame'].min().min(), combined_df['frame'].max().max())

            print(xlims,ylims,tlims)

            df_pre = df_pre[df_pre['frame'] <= frame]
            df_post = df_post[df_post['frame'] <= frame]

        df_pre = df_pre[df_pre['likelihood'] > self.pcutoff].copy()
        df_pre = df_pre.sort_values(by='bodyparts', ascending=True)
        df_post = df_post[df_post['likelihood'] > self.pcutoff].copy()
        df_post = df_post.sort_values(by='bodyparts', ascending=True)

        plotter1 = rp.scatter(df_pre[df_pre['bodyparts'].isin(self.bodyparts)],
                              xlab='x', ylab='frame', zlab='bodyparts', colors=self.palette,darkmode=True)
        plotter2 = rp.scatter(df_pre[df_pre['bodyparts'].isin(self.bodyparts)],
                              xlab='x', ylab='y', zlab='bodyparts', colors=self.palette)
        plotter3 = rp.scatter(df_post[df_post['bodyparts'].isin(self.bodyparts)],
                              xlab='x', ylab='frame', zlab='bodyparts', colors=self.palette)
        plotter4 = rp.scatter(df_post[df_post['bodyparts'].isin(self.bodyparts)],
                              xlab='x', ylab='y', zlab='bodyparts', colors=self.palette)

        if frame is None:sub = rp.subplots(2, 2)
        else: sub = rp.subplots(1,4)

        fig, axes = sub.plot(plotter1, {'title': title + " pre-filtered", 'linewidth': 0, 's': self.s},
                             plotter2, {'title': title + " pre-filtered", 'linewidth': 0, 's': self.s},
                             plotter3, {'title': title + " post-filtered", 'linewidth': 0, 's': self.s},
                             plotter4, {'title': title + " post-filtered", 'linewidth': 0, 's': self.s},
                             figsize=(6, 5), dpi=200, folder_name=self.data_file_name + "_SUB.png")

        if frame is None:
            axes[0,1].get_legend().set_visible(False)
            axes[1,1].get_legend().set_visible(False)
            sub.save()
        else:
            axes[1].get_legend().set_visible(False)
            axes[3].get_legend().set_visible(False)

            for ax in [axes[1], axes[3]]:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
            for ax in [axes[0], axes[2]]:
                ax.set_xlim(*xlims)
                ax.set_ylim(*tlims)

        return fig

    def animator(self):
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        # Function to create a plot
        def create_plot(frame):
            fig = self.plot_generator(frame)
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            cols, rows = fig.canvas.get_width_height()
            img_array = np.frombuffer(buf, dtype=np.uint8).reshape(rows, cols, 4)
            plt.close(fig)
            return img_array[::4,::4,:3]

        cap = cv2.VideoCapture(swap_ext(self.current_path,self.model_name + '_p' + str(int(self.pcutoff*100)) + '_labeled.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create a VideoWriter object
        out = cv2.VideoWriter(swap_ext(self.current_path,'_graph_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                plot_image = create_plot(frame_count)
                plot_height, plot_width, _ = plot_image.shape
                plot_image_bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
                frame[0:plot_height, 0:plot_width] = plot_image_bgr
                out.write(frame)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f'Processed {frame_count} frames')

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# %% INTERNAL METHODS
    def select_initial_files(self):
        root = tk.Tk()
        root.withdraw()
        self.video_paths = filedialog.askopenfilenames(
            title="Select Files", filetypes=[("Video Files", "*.*")],initialdir=self.defaultdir)
        for path in self.video_paths: print("Selected file: ", path)
        root.destroy()
        return

    def simplify_df(self,df):
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
        self.df = pd.DataFrame(self.df)
        try:
            self.df.to_hdf(self.data_file_name + '_backup.h5', key='df', mode='w')
            self.processed_df.to_hdf(self.data_file_name + '.h5', key='df', mode='w')
            print(f"Processed data successfully written for '{self.data_file_name}'.h5.")
        except Exception as e:
            print(f"Error writing: " + self.data_file_name + f".h5 to HDF5 file: {e}")
            return

        # SAVE CSV
        try:
            self.df.to_csv(self.data_file_name + '_backup.csv', index=True)
            self.processed_df.to_csv(self.data_file_name + '.csv', index=True)
            print(f"Processed data successfully saved as CSV to '{self.data_file_name}'.csv.")
        except Exception as e:
            print(f"Error writing:" + self.data_file_name + f".csv to CSV file: {e}")

    def get_text(self):
        def get_text_button():
            self.text = entry.get()
            root.withdraw()
            root.update()
            root.destroy()

        root = tk.Tk()
        root.title("Text Prompt Example")
        entry = tk.Entry(root, width=30)
        entry.pack(pady=10)
        button = tk.Button(root, text="Submit", command=get_text_button)
        button.pack(pady=5)
        root.mainloop()
        return self.text


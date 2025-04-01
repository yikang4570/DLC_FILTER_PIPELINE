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
import matplotlib.pyplot as plt
import readyplot as rp
import tables
from sklearn.cluster import DBSCAN

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
        """Initializes the video pipeline manager. Sets class variables from input_dict, and applies necessary
        changes to the package config.yaml for DLC to run as expected. Forces GUI video selection if none provided.
        :param input_dict: (dict): Resolved dictionary of inputs and defaults from __init__.py"""

        self.input_dict = input_dict
        for name, value in input_dict.items(): setattr(self, name, value)
        self.set_config_pcutoff()
        if 'video_paths' not in input_dict: self.select_initial_files()
        if self.verbose:
            for name, value in self.__dict__.items(): print(f"{name}: {value}")

# %% MAIN METHODS
    def batch_process(self):
        """Processes all provided videos in a loop and keeps track of completed videos in case of failure."""
        self.fully_processed_list = []
        for i, path in enumerate(self.video_paths):
            print(f"Processing {path}")
            self.single_process(path)
            print(f"Finished processing {path}")
            self.fully_processed_list.append(path)

    def single_process(self,path=None):
        """Processes a single video through the entire pipeline. User settings are applied internally to these functions to
        determine whether they will be fully executed.
        :param path: (string): Path of the video to be processed, if None the first video in the list is used."""
        self.initialize_video_path(path)
        self.dlc_analyze()
        self.ensure_model_name()
        self.load_h5()
        self.df_processed = self.custom_filter()
        self.save()
        self.dlc_create_videos()
        self.animator()

    def custom_filter(self):
        if self.verbose: print("Initial DataFrame:", self.df.head())
        print(self.df.columns.names)
        self.processed_df = self.df.copy()
        idx = pd.IndexSlice
        selected_columns = self.processed_df.loc[:, idx[:, self.bodyparts, :]]
        Multi_X = selected_columns.to_numpy()

        def cluster_a_bodypart(MX,n):
            X = MX[:,0+n*3:3+n*3]
            num_rows = X.shape[0]
            new_column = (np.arange(num_rows)).reshape([X.shape[0],1])
            X = np.concatenate((X, new_column), axis=1)

            dbscan = DBSCAN(eps=8, min_samples=8)
            clusters = dbscan.fit_predict(X)

            tab20 = plt.get_cmap('tab20')
            colors = [tab20(i) for i in np.linspace(0, 1, 20)]
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']
            title_suffix = ": " + self.current_path.split(os.sep)[-1]

            combined_array = np.column_stack((X[:,0],X[:,1],X[:,2],X[:,3], clusters))
            tempDF = pd.DataFrame(combined_array, columns=['x','y','p','frame','cluster'])


            temp_plot0 = rp.scatter(tempDF[(tempDF['cluster']>-1) & (tempDF['p']>self.pcutoff)],
                                    xlab='x',ylab='y',zlab='cluster',title=self.bodyparts[n]+title_suffix,
                                    colors=colors,markers=markers,legend=None,darkmode=True)
            temp_plot1 = rp.scatter(tempDF[(tempDF['cluster']>-1) & (tempDF['p']>self.pcutoff)],
                                    xlab='x', ylab='frame', zlab='cluster',title=self.bodyparts[n]+title_suffix,
                                    colors=colors,markers=markers, legend=None,darkmode=True)

            sub = rp.subplots(1,2)
            fig, axes = sub.plot(
                temp_plot0,{'linewidth': 0, 's': self.s},
                temp_plot1,{'linewidth': 0, 's': self.s},
                figsize=(6, 3), dpi=200, folder_name=self.data_file_name + self.bodyparts[n] + ".png")
            for i,ax in enumerate(axes):
                ax.get_legend().set_visible(False)
                ax.set_xlim(0,tempDF['x'].max())
                #ax.set_ylim(0,tempDF['y'].max()) if i == 0 else ax.set_ylim(0,tempDF['frame'].max())

        for n in range((len(self.bodyparts))):
            cluster_a_bodypart(Multi_X,n)

        self.plot_generator()
        if self.verbose: print("Processed DataFrame:", self.df_processed.head())
        return self.processed_df

    def plot_generator(self,frame=None):
        title = self.current_path.split(os.sep)[-1]

        df_pre, df_post = self.simplify_df(self.df), self.simplify_df(self.processed_df)

        if frame is not None:
            combined_df = pd.concat([df_pre, df_post], axis=1)
            xlims = (combined_df['x'].min().min(), combined_df['x'].max().max())
            ylims = (combined_df['y'].min().min(), combined_df['y'].max().max())
            tlims = (combined_df['frame'].min().min(), combined_df['frame'].max().max())

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
        if not self.animate: return

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
    def set_config_pcutoff(self):
        """Set the pcutoff parameter in the config.yaml file to ensure DLC functions behave as expected."""
        self.config_path = os.getcwd() + os.sep + "dlcfilterpipeline" + os.sep + "config.yaml"
        yaml = YAML()
        with open(self.config_path, 'r') as file: data = yaml.load(file)
        data['pcutoff'] = self.pcutoff
        with open(self.config_path, 'w') as file:
            yaml.dump(data, file)

    def select_initial_files(self):
        """GUI multi-file selector to determine list of videos to process if not provided programmatically."""
        root = tk.Tk()
        root.withdraw()
        self.video_paths = filedialog.askopenfilenames(
            title="Select Files", filetypes=[("Video Files", "*.*")],initialdir=self.defaultdir)
        for path in self.video_paths: print("Selected file: ", path)
        root.destroy()
        return

    def initialize_video_path(self,path):
        """Parses whether path is None, int, or string, and assigns self.current_path accordingly. Uses video[0] if None.
        :param path: (None,int,string): Path value to be parsed to determine self.current_path"""
        if path is None:
            print("No explicit video called for single_process(), using first video in the self.video_paths list")
            path = self.video_paths[0]
        elif isinstance(path, int): path = self.video_paths[path]
        self.current_path = path

    def ensure_model_name(self):
        """Determines full name of the model from defaults or GUI user input to manage file naming conventions later."""
        if not hasattr(self, 'model_name'): self.model_name = self.get_text()
        else: print("model_name is already in the class as: ",self.model_name)
        self.data_file_name = swap_ext(self.current_path, self.model_name)

    def dlc_analyze(self):
        """Runs dlc_analyze.py in a DEEPLABCUT virtual environment if user chooses, and writes the model name to json"""

        # PASS IF USER DOES NOT WANT TO ANALYZE
        if not self.analyze_videos: return

        # RUN DLC_ANALYZE.PY IN THE DEEPLABCUT ENVIRONMENT
        subprocess.run([
            self.dlc_venv_path,
            "dlcfilterpipeline/dlc_analyze.py",
            self.config_path,
            self.current_path,
            self.shuffle])

        # SAVE THE CREATED MODEL NAME TO JSON FOR FUTURE USE
        with open("dlcfilterpipeline" + os.sep + "model_name.json", 'r') as openfile:
            json_object = json.load(openfile)
            self.model_name = json_object["model_name"]
            print("CREATED MODEL NAME IS: ", self.model_name)

    def load_h5(self):
        """Loads the h5 file associated with the video as self.df (pandas dataframe)."""
        try:
            self.df = pd.read_hdf(self.data_file_name + '.h5')
            print("HDF5 file: " + self.data_file_name + '.h5' + " successfully loaded.")
        except Exception as e:
            print(f"Error reading HDF5 file:" + self.data_file_name + f" ERROR: {e}")
            return

    def simplify_df(self,df):
        """Simplifies a multi-index data frame to be easier to work with for instance with plotting.
        :param df: Multi-index data frame to be simplified
        :return df_simple: regular dataframe"""

        # SET UP THE COLUMN NAMES AND EXISTING BODYPARTS TO EXTRACT
        cs = ['frame', 'x', 'y', 'likelihood', 'bodyparts']
        scorer = df.columns.get_level_values(0).unique().tolist()[0]
        bodyparts = df.columns.get_level_values(1).unique().tolist()
        coords = df.columns.get_level_values(2).unique().tolist()
        df_simple = pd.DataFrame(columns=cs)

        # ITERATE THROUGH DATA FRAME AND APPEND ROWS PER BODY PART TO THE SIMPLIFIED DF WITH BODY PART AS A VALUE
        for i in range(len(df)):
            tpdf = df.iloc[i]
            for bodypart in bodyparts:
                new_row = [i,
                           tpdf.loc[(scorer, bodypart, cs[1])],
                           tpdf.loc[(scorer, bodypart, cs[2])],
                           tpdf.loc[(scorer, bodypart, cs[3])],
                           bodypart]
                df_simple.loc[len(df_simple)] = new_row

        # RETURN SIMPLIFIED DATAFRAME
        return df_simple

    def save(self):
        """Saves output h5 and csv files and backups of original if user chooses to, required for some future steps."""

        # PASS IF USER DOES NOT WANT TO SAVE
        if not self.save_files:
            print("Files not saving per user settings")
            return

        # SAVE H5
        self.df = pd.DataFrame(self.df)

        try:
            self.df.to_hdf(self.data_file_name + '_backup.h5', key='df', mode='w')
            self.processed_df.to_hdf(self.data_file_name + '.h5', key='df', mode='w')
            print(f"Processed data successfully written for '{self.data_file_name}'.h5.")
        except Exception as e:
            print(f"Error writing: " + self.data_file_name + f".h5 to HDF5 file: {e}")

        # SAVE CSV
        try:
            self.df.to_csv(self.data_file_name + '_backup.csv', index=True)
            self.processed_df.to_csv(self.data_file_name + '.csv', index=True)
            print(f"Processed data successfully saved as CSV to '{self.data_file_name}'.csv.")
        except Exception as e:
            print(f"Error writing:" + self.data_file_name + f".csv to CSV file: {e}")

    def dlc_create_videos(self):
        """Runs dlc_create_video.py in a DEEPLABCUT virtual environment if user chooses"""

        # PASS IF USER DOES NOT WANT TO CREATE VIDEOS
        if not self.create_videos: return

        # START VIRTUAL ENVIRONMENT TO RUN DLC VIDEO CREATION
        subprocess.run([
            self.dlc_venv_path,
            "dlcfilterpipeline/dlc_create_video.py",
            self.config_path,
            self.current_path,
            self.shuffle,
            ",".join(self.bodyparts)])

    def get_text(self):
        """GUI for user text acquisition
        :return self.text: (string): Text entered by the user"""
        # INITIALIZE BUTTON
        def get_text_button():
            self.text = entry.get()
            root.withdraw()
            root.update()
            root.destroy()

        # TKINTER MAINLOOP WHICH CATCHES TEXT SUBMISSION
        root = tk.Tk()
        root.title("Text Prompt Example")
        entry = tk.Entry(root, width=30)
        entry.pack(pady=10)
        button = tk.Button(root, text="Submit", command=get_text_button)
        button.pack(pady=5)
        root.mainloop()
        return self.text


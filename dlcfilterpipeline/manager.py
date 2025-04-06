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
from scipy.ndimage import median_filter

# Directory management
import os
import tkinter as tk
from tkinter import filedialog
import subprocess
import json
from ruamel.yaml import YAML
from pathlib import Path
import scipy.io

# ML functions
from sklearn.cluster import DBSCAN

# Package utils
from .utils import (dict_update_nested,
                    mini_kwarg_resolver,
                    swap_ext,
                    between_fill,
                    trim_df_by_x)

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
        self.initialize_cluster_colormap()
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
        self.custom_filter()
        self.save()
        self.dlc_create_videos()
        self.animator()
        self.create_AGATHA_arrays()
        self.save_to_mat()

    def custom_filter(self):
        Multi_X = self.initialize_processed_df_numpy()

        for n in range((len(self.bodyparts))):
            self.filter_a_limb(Multi_X,n)
        self.filter_nose_to_tail()

        self.plot_generator()
        if self.verbose: print("Processed DataFrame:", self.df_processed.head())
        return

    def initialize_processed_df_numpy(self):
        """Initializes processed_df, and extracts a simplified numpy array Multi_x for cluster processing
        :return Multi_X: (np.array): Input numpy array of all tracking data"""

        # PRINT INITIAL INFO
        if self.verbose: print("Initial DataFrame:", self.df.head())
        print(self.df.columns.names)

        # CREATE PROCESSED_DF
        self.processed_df = self.df.copy()

        # CREATE SIMPLIFIED ARRAY VERSION TO PREP CLUSTERING
        idx = pd.IndexSlice
        selected_columns = self.processed_df.loc[:, idx[:, self.bodyparts, :]]
        Multi_X = selected_columns.to_numpy()

        # RETURN NEW ARRAY
        return Multi_X

    def filter_a_limb(self,MX, n):
        """Clusters limb footsteps, initializes single readyplots, and plots a layout (readyplot subplot)
        :param MX: (np.array): Input numpy array of all tracking data
        :param n : (int): Number of current body part"""

        # GET TEMPORARY DATA FRAME AND COPY THE P VALUES TO THE ORIGINAL
        tempDF = self.extract_and_cluster_limb(MX, n)
        self.processed_df.loc[:, (slice(None), self.bodyparts[n], 'x')] = tempDF['x']
        self.processed_df.loc[:, (slice(None), self.bodyparts[n], 'y')] = tempDF['y']
        self.processed_df.loc[:, (slice(None), self.bodyparts[n], 'likelihood')] = tempDF['p']

        # CREATE PLOTS
        temp_plot0,temp_plot1 = self.cluster_plot_singles(tempDF,n)
        self.cluster_plot_layout(tempDF,n,temp_plot0,temp_plot1)

    def filter_nose_to_tail(self):
        def smooth_and_fill(n,k):
            mask_constructor = self.processed_df.loc[:,(slice(None),self.ref_bodyparts[n],'likelihood')]<self.pcutoff
            mask = mask_constructor.to_numpy().ravel()
            tuple_indx = (slice(None), self.ref_bodyparts[n], k)

            self.processed_df.loc[mask,tuple_indx] = np.nan
            self.processed_df.loc[:,tuple_indx] = self.processed_df.loc[:,tuple_indx].ffill().bfill()

            self.processed_df.loc[:, tuple_indx] = median_filter(
                self.processed_df.loc[:, tuple_indx], size = 5)

        for n in [0,1]:
            for k in ['x','y']:
                smooth_and_fill(n,k)
            # SET ALL LIKELIHOODS TO 1 TO AVOID FILTERING OUT LATER
            self.processed_df.loc[:, (slice(None), self.ref_bodyparts[n], 'likelihood')] = 1
            self.nose_tail_plot(n)

    def extract_and_cluster_limb(self, MX, n):
        """Unpacks the numpy array by body part (indicated by n), clusters footsteps, and returns a convenient tempDF
        :param MX: (np.array): Input numpy array of all tracking data
        :param n: (int): Number of current body part
        :returns: tempDF: (DataFrame): Temporary data relevant to the current body part"""

        # EXTRACT DESIRED COLUMNS FROM MX
        X = MX[:, 0 + n * 3:3 + n * 3]
        num_rows = X.shape[0]
        new_column = (np.arange(num_rows)).reshape([X.shape[0], 1])
        X = np.concatenate((X, new_column), axis=1)

        # CLUSTER FOOTSTEPS
        dbscan = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_samples)
        clusters = dbscan.fit_predict(X)

        # CREATE TEMPORARY DATAFRAME AND SET UN-CLUSTERED P VALUES TO 0, FILL VALUES WITHIN CLUSTERS
        combined_array = np.column_stack((X[:, 0], X[:, 1], X[:, 2], X[:, 3], clusters))
        tempDF = pd.DataFrame(combined_array, columns=['x', 'y', 'p', 'frame', 'cluster'])

        tempDF.loc[tempDF['p'] <= self.pcutoff,'cluster'] = -1
        minFrame,maxFrame = tempDF['frame'].min(), tempDF['frame'].max()
        average_cluster_size = len(tempDF)/len(tempDF['cluster'].unique())
        cluster_trim = int(average_cluster_size * 0.01 * self.cluster_trim_percent)
        cluster_exclude = int(average_cluster_size * 0.01 * self.cluster_exclusion_percent)
        print("Cluster trimmed by the following at beginning and end:", cluster_trim)

        tempDF = between_fill(tempDF,'cluster')
        for cluster in tempDF['cluster'].unique():
            print('Processing ' + self.bodyparts[n] + ' cluster #:' + str(cluster),
                'Length of cluster: ' + str(len(tempDF[tempDF['cluster'] == cluster])),
                  'Excluding if less than: ' + str(cluster_exclude))
            if len(tempDF[tempDF['cluster'] == cluster]) < cluster_exclude:
                tempDF.loc[tempDF['cluster'] == cluster,'cluster'] = -1
                print('Excluded cluster #: ' + str(cluster) + ' because it was under cluster_exclusion_percent for foot')
            elif tempDF.loc[tempDF['cluster'] == cluster,'frame'].isin([minFrame,maxFrame]).any() :
                tempDF.loc[tempDF['cluster'] == cluster, 'cluster'] = -1
                print('Excluded cluster #: ' + str(cluster) + ' because it borders beginning or end of movie')
            else:
                temp_mean = tempDF[tempDF['cluster'] == cluster]['p'].mean()
                tempDF.loc[tempDF['cluster'] == cluster,'p'] = temp_mean
                trimmed_list = trim_df_by_x(tempDF[tempDF['cluster'] == cluster]['p'].to_list(),cluster_trim)
                tempDF.loc[tempDF['cluster'] == cluster,'p'] = trimmed_list
                tempDF.loc[tempDF['cluster'] == cluster,'x'] = median_filter(tempDF.loc[tempDF['cluster'] == cluster,'x'],size=5)
                tempDF.loc[tempDF['cluster'] == cluster,'y'] = median_filter(tempDF.loc[tempDF['cluster'] == cluster,'y'],size=5)

        tempDF.loc[tempDF['cluster'] == -1,'p'] = 0.0

        # RETURN SUBSET ARRAY, CLUSTERS, AND EDITED TEMPDF
        return tempDF.astype('float32')

    def cluster_plot_singles(self, tempDF,n, ylabs = ['y','frame']):
        """Loops through provided ylabs to create cluster plots
        :param tempDF: (DataFrame): Dataframe containing columns 'x', 'y', 'p', and 'frame'
        :param n: (int): Current bodypart number
        :param ylabs: (list of strings): List of ylab names for plotting
        :return: temp_plots.values(): (list): readyplot objects to use in subplots later, unpack with rp0, rp1 = fnc()"""

        # FILTER TEMPDF BY PCUTOFF FOR PLOTTING, INITIALIZE OUTPUT DICT
        plotDF = tempDF[tempDF['p'] > self.pcutoff]
        temp_plots = {}

        # KWARGS FOR READYPLOT, MOSTLY ESTHETICS BUT ALSO TITLE AND AVOIDING ERRORS WITH MARKER INPUT
        cluster_ptk = {'title':self.bodyparts[n] + ": " + self.current_path.split(os.sep)[-1],
            'colors':self.cluster_colors, 'markers':self.cluster_markers, 'legend':None, 'darkmode':True}

        # ITERATIVELY INITIALIZE READYPLOT OBJECTS AND RETURN AN UNPACKABLE LIST OF THEM
        for ylab in ylabs: temp_plots[ylab] = rp.scatter(plotDF, xlab='x', ylab=ylab, zlab='cluster',**cluster_ptk)
        return temp_plots.values()

    def cluster_plot_layout(self,tempDF,n,temp_plot0,temp_plot1):
        # ARRANGE SUBPLOTS AND APPLY EXTRA SETTINGS
        sub = rp.subplots(1, 2)
        fig, axes = sub.plot(
            temp_plot0, {'linewidth': 0, 's': self.s},
            temp_plot1, {'linewidth': 0, 's': self.s},
            figsize=(6, 3), dpi=200, folder_name=self.data_file_name + self.bodyparts[n] + ".png", save=False)
        for i, ax in enumerate(axes):
            ax.get_legend().set_visible(False)
            ax.set_xlim(0, tempDF['x'].max())
            # ax.set_ylim(0,tempDF['y'].max()) if i == 0 else ax.set_ylim(0,tempDF['frame'].max())
        sub.save()

    def initialize_cluster_colormap(self):
        """Provides a large (n=20) colormap for cluster footstep plotting unless colors have already been provided."""
        if hasattr(self, 'cluster_colors'): return
        tab20 = plt.get_cmap('tab20')
        self.cluster_colors = [tab20(i) for i in np.linspace(0, 1, 20)]

    def nose_tail_plot(self,n):
        extra_plot_kwargs = {'title': self.ref_bodyparts[n] + ": " + self.current_path.split(os.sep)[-1],
                       'colors': self.cluster_colors, 'markers': self.cluster_markers, 'legend': None, 'darkmode': True}

        prex = self.df.loc[:, (slice(None), self.ref_bodyparts[n], 'x')].to_numpy().ravel()
        postx = self.processed_df.loc[:, (slice(None), self.ref_bodyparts[n], 'x')].to_numpy().ravel()
        prey = self.df.loc[:, (slice(None), self.ref_bodyparts[n], 'y')].to_numpy().ravel()
        posty = self.processed_df.loc[:, (slice(None), self.ref_bodyparts[n], 'y')].to_numpy().ravel()
        pret = self.df.index.tolist()
        postt = self.processed_df.index.tolist()

        extra_plot_kwargs['title'] = extra_plot_kwargs['title'] + '_pre'
        plot0 = rp.scatter(prex,prey, xlab='x', ylab='y', zlab='cluster', **extra_plot_kwargs)
        plot1 = rp.scatter(prex,pret, xlab='x', ylab='t', zlab='cluster', **extra_plot_kwargs)
        extra_plot_kwargs['title'] = extra_plot_kwargs['title'][:-4] + '_post'
        plot2 = rp.scatter(postx,posty, xlab='x', ylab='y', zlab='cluster', **extra_plot_kwargs)
        plot3 = rp.scatter(postx,postt, xlab='x', ylab='t', zlab='cluster', **extra_plot_kwargs)

        sub = rp.subplots(2, 2)
        fig, axes = sub.plot(
            plot0, {'linewidth': 0, 's': self.s},
            plot1, {'linewidth': 0, 's': self.s},
            plot2, {'linewidth': 0, 's': self.s},
            plot3, {'linewidth': 0, 's': self.s},
            figsize=(6, 3), dpi=200, folder_name=self.data_file_name + self.ref_bodyparts[n] + ".png", save=False)

        for i in range(2):
            for j in range(2): axes[i,j].get_legend().set_visible(False)

        sub.save()

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

            if frame_count % self.graph_video_frame_step == 0:
                plot_image = create_plot(frame_count)
                plot_height, plot_width, _ = plot_image.shape
                plot_image_bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
                frame[0:plot_height, int((frame_width-plot_width)/2):int((frame_width-plot_width)/2)+plot_width] = plot_image_bgr
                out.write(frame)

            frame_count += 1
            if frame_count % 10 == 0:
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
        :return: df_simple: regular dataframe"""

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

        # SAVE DIRECTORY
        directory, backup_filename = os.path.split(self.data_file_name)
        backup_directory = directory + os.sep + 'BACKUPS'
        os.makedirs(backup_directory, exist_ok=True)
        backup_filepath = str(os.path.join(backup_directory, backup_filename))

        # SAVE H5
        self.df = pd.DataFrame(self.df)

        try:
            self.df.to_hdf(backup_filepath + '.h5', key='df', mode='w')
            self.processed_df.to_hdf(self.data_file_name + '.h5', key='df', mode='w')
            print(f"Processed data successfully written for '{self.data_file_name}'.h5.")
        except Exception as e:
            print(f"Error writing: " + self.data_file_name + f".h5 to HDF5 file: {e}")

        # SAVE CSV
        try:
            self.df.to_csv(backup_filepath + '.csv', index=True)
            self.processed_df.to_csv(self.data_file_name + '.csv', index=True)
            print(f"Processed data successfully saved as CSV to '{self.data_file_name}'.csv.")
        except Exception as e:
            print(f"Error writing:" + self.data_file_name + f".csv to CSV file: {e}")

    def dlc_create_videos(self):
        """Runs dlc_create_video.py in a DEEPLABCUT virtual environment if user chooses"""

        # PASS IF USER DOES NOT WANT TO CREATE VIDEOS
        if not self.create_videos: return

        # START VIRTUAL ENVIRONMENT TO RUN DLC VIDEO CREATION
        label_list = self.bodyparts.copy()
        label_list.extend(self.ref_bodyparts)
        print("Creating labeled videos of bodyparts: ", label_list)
        subprocess.run([
            self.dlc_venv_path,
            "dlcfilterpipeline/dlc_create_video.py",
            self.config_path,
            self.current_path,
            self.shuffle,
            ",".join(label_list)])

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

    def create_AGATHA_arrays(self):
        self.nose_array = np.array([
            self.processed_df.loc[:, (slice(None), self.ref_bodyparts[0], 'x')].to_numpy().ravel(),
            self.processed_df.loc[:, (slice(None), self.ref_bodyparts[0], 'y')].to_numpy().ravel()]).T

        self.tail_array = np.array([
            self.processed_df.loc[:, (slice(None), self.ref_bodyparts[1], 'x')].to_numpy().ravel(),
            self.processed_df.loc[:, (slice(None), self.ref_bodyparts[1], 'y')].to_numpy().ravel()]).T
        self.centroid_array = (self.nose_array + self.tail_array)/2

        self.AGATHA_rows = []
        for bodypart in self.bodyparts: self.isolate_footstrikes(bodypart)
        self.AGATHA_array = np.array(self.AGATHA_rows)
        self.AGATHA_array = self.AGATHA_array[self.AGATHA_array[:, 3].argsort()]
        for i in range(self.AGATHA_array.shape[0]): self.AGATHA_array[i][0] = i+1

    def isolate_footstrikes(self,bp):
        x = self.processed_df.loc[:, (slice(None), bp, 'x')].to_numpy().ravel()
        y = self.processed_df.loc[:, (slice(None), bp, 'y')].to_numpy().ravel()
        p = self.processed_df.loc[:, (slice(None), bp, 'likelihood')].to_numpy().ravel()

        i,start_indices,end_indices = 1,[],[]

        while i < len(p):
            if p[i] != 0 and p[i-1] == 0: start_indices.append(i)
            elif p[i] == 0 and p[i-1] != 0: end_indices.append(i)
            i += 1

        fore_or_hind = 1 if 'F' in bp else 0
        left_or_right = 1 if 'L' in bp else 0
        mouse_dir = 1 if x[-1] < x[0] else 0

        for i in range(len(start_indices)):
            AGATHA_row = [0,# Temporary Step Number, filter later by T, start at 1
                          fore_or_hind,# Fore or Hind
                          int(x[start_indices[i]]),# X Foot-strike
                          start_indices[i],# T Foot-strike
                          int(x[end_indices[i]]),# X Toe-off
                          end_indices[i],# T Toe-off
                          np.average(np.array(x[start_indices[i]:end_indices[i]])),# X Centroid
                          int(start_indices[i] + end_indices[i])/2,# T Centroid
                          left_or_right,# Left or Right
                          x[start_indices[i]],# X Foot-strike
                          y[start_indices[i]],# Y Foot-strike
                          mouse_dir]# Mouse Dir

            self.AGATHA_rows.append(AGATHA_row)

        return

    def save_to_mat(self):
        nested_dict = {
            'DATA': {
                'Velocity': {
                    'BottomCentroidVelocity': self.centroid_array,
                    'BottomNoseVelocity': self.nose_array,
                    'TopCentroidVelocity': self.centroid_array,
                    'TopNoseVelocity': self.nose_array
                },
                'AGATHA': self.AGATHA_array
            }
        }

        scipy.io.savemat(self.data_file_name + '_DATA.mat', nested_dict)
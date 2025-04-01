#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An initialization file
@author: Shawn Pavey
"""
# DLC_FILTER_PIPELINE/__init__.py
# %% IMPORT PACKAGES
# Data packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Directory management
import os

# Class nickname import
from .manager import Manager

# Package utils
from .utils import dict_update_nested

#%%---------------------------------------------------------------------------------------------------------------------
# MAIN INITIALIZATION
#-----------------------------------------------------------------------------------------------------------------------
# %% PARSE ARGS AND PREPARE DATA FRAME COLUMN NAMES TO HANDLE ANY COMBINATION OF STRING INPUTS (OR NON-INPUTS)
def pre_init(*input_args,**input_kwargs):
    # INITIALIZE DEFAULT VALUES
    defaults_dict = {
        'pcutoff':0.7,
        'bodyparts': ['FL_ground', 'FR_ground', 'BL_ground', 'BR_ground'],
        'defaultdir': os.getcwd(),
        'analyze_videos': True,
        'create_videos': True,
        'animate': False,
        'shuffle': '0',
        'save_files': True,
        'dlc_venv_path' : "/opt/anaconda3/envs/DEEPLABCUT/bin/python",
        'verbose': False,
        'palette': ['gold', 'red', 'cyan', 'lime'],
        's': 10
    }

    # START INPUT_DICT FROM SUPPLIED KWARGS
    input_dict = input_kwargs

    # RESOlVE ANY INPUT ARGS INTO THE INPUT DICT
    if len(input_args) == 0: pass
    elif len(input_args) == 1 and isinstance(input_args,list): input_dict['video_paths'] = input_args[0]
    elif len(input_args) == 1 and not isinstance(input_args,list): input_dict['video_paths'] = [input_args[0]]
    else: input_dict['video_paths'] = list(input_args)

    # RESOLVE INPUTS WITH DEFAULTS
    input_dict = dict_update_nested(defaults_dict,input_dict)

    return input_dict

#%%---------------------------------------------------------------------------------------------------------------------
# EXTRA INIT METHODS
#-----------------------------------------------------------------------------------------------------------------------


#%%---------------------------------------------------------------------------------------------------------------------
# ROUTING MANAGER CLASS
#-----------------------------------------------------------------------------------------------------------------------
# %% MANAGER
def manager(*args,**kwargs):
    input_dict = pre_init(*args,**kwargs)
    return Manager(input_dict)

# %% EXPLICITLY STATE HOW TO IMPORT THE ENTIRE MODULE (eg: import *)
__all__ = ["manager", "Manager"]

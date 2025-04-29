#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A utility file, utils, which holds many common small functions that are often mathematical/do not use 'self'
@author: Shawn Pavey
"""
# %% IMPORTS
import os
import tkinter as tk
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

# %% FUNCTIONS
def mini_kwarg_resolver(key,def_val,kwargs):
    """Resolves kwargs specified by the key with default values"""
    if key not in kwargs:
        output = def_val
    else:
        output = kwargs[key]
        del kwargs[key]
    return output, kwargs

def dict_update_nested(default_dictionary, input_dictionary):
    """Recursively update a nested dictionary"""
    for key, value in input_dictionary.items():
        if isinstance(value, dict):
            if key in default_dictionary:
                default_dictionary[key] = dict_update_nested(default_dictionary[key], value)
            else:
                default_dictionary[key] = value
        else:
            default_dictionary[key] = value
    return default_dictionary

def swap_ext(filename, new_extension):
    """Swaps filename extension with new_extension"""
    return os.path.splitext(filename)[0] + new_extension

def between_fill(df,col_name):
    """Takes a DataFrame and combines forward and back filling. Where both agree, the new value is used (fills clusters)"""
    forward_fill_df = df[[col_name]].copy().replace(-1,np.nan).ffill()
    back_fill_df = df[[col_name]].copy().replace(-1, np.nan).bfill()
    mask = forward_fill_df == back_fill_df
    result = pd.DataFrame(-1, index=forward_fill_df.index, columns=forward_fill_df.columns)
    result[mask] = forward_fill_df[mask]
    differences = result - df[col_name]
    print('Number of filled rows: ',differences.sum().sum())
    df[col_name] = result
    return df


def trim_df_by_x(lst, x):
    if x > 0:
        num_elements_to_zero = min(x, len(lst))
        lst[:num_elements_to_zero] = [0] * num_elements_to_zero
        lst[-num_elements_to_zero:] = [0] * num_elements_to_zero
    else:
        pass
    return lst

def d_dt(x):
    dx_dt = median_filter(x.copy(),size=5)
    for i in range(len (x)):
        if i == 0:
            dx_dt[i] = (x[1]+x[2]-2*x[0])/2
        elif i == len(x)-1:
            dx_dt[i] = (2*x[-1]-x[-2]-x[-3])/2
        else:
            dx_dt[i] = ((x[i+1]-x[i]) + (x[i]-x[i-1]))/2
    return dx_dt


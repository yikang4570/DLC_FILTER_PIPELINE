#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A utility file, utils, which holds many common small functions that are often mathematical/do not use 'self'
@author: Shawn Pavey
"""
# %% IMPORTS
import os
import tkinter as tk

# %% FUNCTIONS
def mini_kwarg_resolver(key,def_val,kwargs):
    if key not in kwargs:
        output = def_val
    else:
        output = kwargs[key]
        del kwargs[key]
    return output, kwargs

def dict_update_nested(default_dictionary, input_dictionary):
    """
    Recursively update a nested dictionary
    """
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
    """
    Swaps filename extension with new_extension
    """
    return os.path.splitext(filename)[0] + new_extension

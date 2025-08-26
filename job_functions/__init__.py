#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An initialization file
@author: Shawn Pavey
"""
# job_functions/__init__.py
# %% IMPORT PACKAGES
from . import failure_logger as _failure_logger
from . import aggregate_sub_files as _aggregate_sub_files

def failure_logger(*args,**kwargs):
    return _failure_logger.main(*args,**kwargs)

def aggregate_sub_files(*args,**kwargs):
    return _aggregate_sub_files.main(*args,**kwargs)

# %% EXPLICITLY STATE HOW TO IMPORT THE ENTIRE MODULE (eg: import *)
__all__ = ["failure_logger", "aggregate_sub_files"]

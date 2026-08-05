#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python program to extract wave parameters from a single signal using InSyncPy class.

Antoine Fortuné
Anastasia MARECHAL
Mathieu MEZACHE
"""

#######################################################################################################################
import os
import sys
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions_InSync as sinc
####################################################################################################

# #########################
# FUNCTIONS
# #########################

def load_signal(filename):
    """
    Load time-signal data from a CSV file.

    Expects a CSV with a header row followed by rows of comma-separated
    time and signal values. Returns two NumPy arrays: time and signal.

    Parameters:
        filename : str or Path
            Path to the CSV file.

    Returns:
        time : np.ndarray
            1D array of time values (string dtype from CSV).
        signal_data : np.ndarray
            1D array of signal values (float64).
    """
    
    if not os.path.isfile(filename):
        print(f("Error: %s is not a valid file", filename))
        exit(1)

    if not filename.endswith('.csv'):
        print(f"Error: CSV file required, with .csv extension. Given : {filename} ." )
        exit(1)

    sig_name = os.path.splitext(filename)[0]

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        data = list(reader)
    time, signal_data = zip(*data)  # Transpose the data to get columns instead of rows
    
    return sig_name, np.array(time), np.array(signal_data, dtype=float)

def times_to_duration(t):
    """
    Convert sequential HH:MM timestamps to cumulative elapsed hours.

    Handles day transitions: e.g., 23:00 -> 01:00 is treated as 2 hours elapsed.
    For time series spanning multiple days, modular arithmetic correctly computes
    elapsed time across any number of day boundaries.

    Args:
        t (numpy.ndarray): Array of time strings in '%H:%M' format (e.g., '23:00', '01:00').

    Returns:
        numpy.ndarray: Cumulative elapsed time in hours, starting from 0.
                       Returns an empty array of dtype float if input is empty.
    """

    if not t.any():
        return np.array([], dtype=float)

    # Vectorized conversion to seconds
    seconds = np.array([
        datetime.strptime(ts, '%H:%M').hour * 3600 + 
        datetime.strptime(ts, '%H:%M').minute * 60 
        for ts in t
    ], dtype=float)

    # Calculate differences between consecutive times
    diffs = np.diff(seconds)

    # Handle any number of day wraparounds using modular arithmetic
    diffs = np.mod(diffs, 86400)

    # Cumulative sum and convert to hours
    time_series = np.insert(np.cumsum(diffs), 0, 0)

    return time_series / 3600

def _is_time_format(arr):
    """
    Check if array contains HH:MM time format strings.
    
    Returns True if all elements match the HH:MM pattern (e.g., '08:30', '23:59').
    Returns False if elements appear to be numeric durations.
    
    Args:
        arr (numpy.ndarray): Array of strings from CSV.
    
    Returns:
        bool: True if time format, False if duration format.
    """
    import re
    # Check first few non-empty elements for HH:MM pattern
    time_pattern = re.compile(r'^\d{2}:\d{2}$')
    samples = [str(x).strip() for x in arr[:10] if str(x).strip()]
    if not samples:
        return False
    
    return all(time_pattern.match(s) for s in samples)


def _is_seconds_format(arr):
    """
    Check if array contains numeric duration values (positive integers).
    
    Attempts to convert elements to float and checks they are non-negative.
    
    Args:
        arr (numpy.ndarray): Array of strings from CSV.
    
    Returns:
        bool: True if all elements can be parsed as non-negative floats.
    """
    samples = [str(x).strip() for x in arr[:10] if str(x).strip()]
    if not samples:
        return False
    
    try:
        values = [int(s) for s in samples]
        return all(v >= 0 for v in values)
    except (ValueError, TypeError):
        return False


def _is_hours_format(arr):
    """
    Check if array contains numeric duration values (positive floats).
    
    Attempts to convert elements to float and checks they are non-negative.
    
    Args:
        arr (numpy.ndarray): Array of strings from CSV.
    
    Returns:
        bool: True if all elements can be parsed as non-negative floats.
    """
    samples = [str(x).strip() for x in arr[:10] if str(x).strip()]
    if not samples:
        return False
    
    try:
        values = [float(s) for s in samples]
        return all(v >= 0 for v in values)
    except (ValueError, TypeError):
        return False


def detect_time_format(time_str_arr):
    """
    Detect whether the input time column contains HH:MM timestamps or numeric durations.
    
    Checks the format of time strings and returns a string indicating the detected format.
    
    Args:
        time_str_arr (numpy.ndarray): Array of time strings from CSV.
    
    Returns:
        str: 'hh:mm' if time format detected, 'duration' if duration format detected,
             'unknown' if format cannot be determined.
    """
    if _is_time_format(time_str_arr):
        return 'hh:mm'
    elif _is_int_format(time_str_arr):
        return 'seconds'
    elif _is_float_format(time_str_arr):
        return 'hours'
    else:
        return 'unknown'


def time_to_hours(time_str_arr):
    """
    Convert an array of time strings to hours as float.

    Args:
       time_str_arr (numpy.ndarray): Array of time

    Returns:
       numpy.ndarray: Array of time in hours as float.
    """
    time_format = detect_time_format(time_str_arr)

    if time_format == 'hours':
        # Convert time (hh:mm) to duration in HOURS (float).
        return time_str_arr

    # Convert time (seconds) to duration in HOURS (float).
    if time_format == 'seconds':
        # Convert time (hh:mm) to duration in HOURS (float).
        print(f"Detected time in seconds. Converted to hours.")
        return time_str_arr.astype(float) / 3600

    # Convert time (hh:mm) to duration in HOURS (float).
    if time_format == 'hh:mm':
        # Convert time (hh:mm) to duration in HOURS (float).
        print(f"Detected HH:MM time format. Converted to hours.")
        return times_to_duration(time_str_arr)

    if time_format == 'unknown':
        print(f"Error: unsupported time format. Please use HH:MM or hours (float).")
        exit(1)


# ########################
# START
# #########################

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} <signal_file.csv>")
    sys.exit(1)

# Load the signal from CSV file
# The first line must be a header
# The first column is the time (HH:mm or hours (float)), the second column is the signal value (float)
sig_name, time, sig = load_signal(sys.argv[1])

t = time_to_hours(time)

# Signal modelling using InSyncPy class
model = sinc.InSyncPy(
    t = t, sig = sig, sig_name = sig_name, 
    show_plot = False, save_plot = False, 
    save_csv = True, dir_save = "example2_out")
# Analyse the signal 
model.full_analysis()

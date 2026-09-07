#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A python script to recursively explore a directory (dataset) and 
to extract the oscillation parameters from any csv file, using InSyncPy class. 
The csv file is assumed to contain the time series (unit: hour) 
and the amplitude of an oscillating signal.

The outputs are written in the output directory (default: ./insync_out/). 
Sub-directories organisation from the source directory is preserved 
in the output_dir/.

iBV - Antoine Fortuné
"""

#######################################################################################################################
import os
import shutil
import sys
import csv
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import functions_InSync as sinc
####################################################################################################

# set data_dir with the first argument of the command
data_dir = sys.argv[1]
output_dir = "./insync_out"

####################################################################################################

# #########################
# FUNCTIONS
# #########################

def _times_str_to_hours(t):
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
    hours = np.array([int(ts.split(':')[0]) * 3600 + int(ts.split(':')[1]) * 60 
                      for ts in t], dtype=float)

    # Calculate differences between consecutive times
    diffs = np.diff(hours)

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

def _is_int_format(arr):
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

def _is_float_format(arr):
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

def _detect_time_format(time_str_arr):
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

def time_as_hours(time_str_arr):
    """
    Convert an array of time strings to hours as float.

    Args:
       time_str_arr (numpy.ndarray): Array of time

    Returns:
       numpy.ndarray: Array of time in hours as float.
    """
    time_format = _detect_time_format(time_str_arr)

    if time_format == 'hours':
        # Convert time (hh:mm) to duration in HOURS (float).
        print(f"Detected time as float, asuming to be in hours.")
        return time_str_arr

    # Convert time (seconds) to duration in HOURS (float).
    if time_format == 'seconds':
        # Convert time (hh:mm) to duration in HOURS (float).
        print(f"Detected time in seconds (int). Converted to hours.")
        return time_str_arr.astype(float) / 3600

    # Convert time (hh:mm) to duration in HOURS (float).
    if time_format == 'hh:mm':
        # Convert time (hh:mm) to duration in HOURS (float).
        print(f"Detected HH:MM time format. Converted to hours.")
        return _times_str_to_hours(time_str_arr)

    if time_format == 'unknown':
        print(f"Error: unsupported time format. Please use HH:MM or hours (float).")
        exit(1)

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

    sig_name = os.path.basename(os.path.splitext(filename)[0])

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header line
        data = list(reader)
    time, signal_data = zip(*data)  # Transpose the data to get columns instead of rows
    
    return sig_name, np.array(time), np.array(signal_data, dtype=float)

def sanitize_lc_signame(name: str):
    """
    Remove lumicycler preprocess suffixes from filenames if exists.
    
    Args:
        name (str): Input string, e.g. 'LANEA4_Bmal1_61W_Raw_boat-in'
    
    Returns:
        str: name with '_boat-in' and '_Raw' suffix removed, e.g. 'LANEA4_Bmal1_61W'
    """
    suffixes = ["_boat-in","_Raw"]
    for suffix in suffixes:
        name = name.replace(suffix, "")
    return name

def recat_metrics_files(output_dir: str):
    """
    Concatenate all *_metrics.csv files in output_dir into a single metrics_sumup.csv.
    
    Args:
        output_dir (str): Root directory containing the processed results.
    """
    metrics_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('_metrics.csv'):
                metrics_files.append(os.path.join(root, file))
    
    if not metrics_files:
        print("No metrics files found. Skipping concatenation.")
        return
    
    # Read the first file to get the header
    with open(metrics_files[0], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        first_data = list(reader)
    
    # Write the concatenated file
    output_file = os.path.join(output_dir, 'metrics_sumup.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(first_data)
        
        # Append data from remaining files
        for metrics_file in metrics_files[1:]:
            with open(metrics_file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                writer.writerows(reader)
    
    print(f"Concatenated {len(metrics_files)} metrics files into {output_file}")

def recurcive_csv_processing(source_path, dest_path):
    print(f"Parsing dir : {source_path}")

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    for file in os.listdir(source_path):
        file_path = os.path.join(source_path, file)
        if os.path.isfile(file_path) and file.endswith('.csv'):
            
            print(f"Processing file: {file}")
            # Load the signal from CSV file
            # The first line must be a header
            # The first column is the time (HH:mm), the second column is the signal (float)
            sig_name, time, sig = load_signal(file_path)
            sig_name = sanitize_lc_signame(sig_name)
            print(f"Sig_name: {sig_name}")
            
            # Convert time (hh:mm) to HOURS (float).
            t = time_as_hours(time)
            #t = time # if time is already in hours

            # creer un objet InSyncPy avec le signal
            signal = sinc.InSyncPy(t = t, sig = sig, sig_name = sig_name)
            # lancer l'analyse complète
            signal.full_analysis(
                show_plot = False, 
                save_plot = True, 
                save_csv = True, 
                dir_save = dest_path
                )
             
        elif os.path.isdir(file_path):
            subdir_name = os.path.basename(file_path)
            recurcive_csv_processing(file_path, os.path.join(dest_path, subdir_name))


# ########################
# START
# #########################

def main():
    """
    Main entry point with proper memory management.
    """
    global data_dir, output_dir

    start_time = datetime.now()
    print(f"insyncpy-pipeline started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset path : {data_dir}")

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <dataset_dir>")
        sys.exit(1)

    # test data_dir is a valid directory
    if not os.path.isdir(data_dir):
        print("Error: data_dir is not a valid directory")
        sys.exit(1)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    try:

        recurcive_csv_processing(source_path=data_dir, dest_path=output_dir)

        # Concatenate all metrics files
        recat_metrics_files(output_dir)

        end_time = datetime.now()
        duration = end_time - start_time
        print("\n -----------------------------")
        print(f"Results saved in {output_dir}")
        print(f"Execution completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {duration}")
        print("-----------------------------")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
    
    return 0

if __name__ == "__main__":
    exit(main())

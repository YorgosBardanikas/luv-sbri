"""Preprocessing script for the LUV project data.
It parses the behavioral and task data and align the MUA on epochs
around the selected event onset. 
It saves a file per session with the MUA epochs and the metadata."""

import os
import utils
import pickle
import numpy as np
import xarray as xr
from parse_sessions import parse_session
from utils import path_mua, sessions_to_drop

event = 'mo'
session_group = sorted(os.listdir(path_mua))

for session in session_group:

    if session in sessions_to_drop: continue
    session_dict = parse_session(session)
    fb_codes = session_dict['fb_codes']
    fb_times = session_dict['fb_times']
    mo_times = session_dict['mo_times']
    target_codes = session_dict['target_codes']
    best_target = session_dict['best_target']
    trials_in_blocks = session_dict['trials_in_blocks']
    
    # Get the relevant codes based on the selected event onset 
    if event == 'fb': ev_times, ev_codes = fb_times, fb_codes
    elif event == 'mo': ev_times, ev_codes = mo_times, target_codes
    else: raise ValueError(f"Unknown event '{event}'")

    # Load the MUA pickle file
    mua_filename = f'{session}_MUAe.pkl'
    with open(os.path.join(path_mua, mua_filename),'rb') as handle:
        mua = pickle.load(handle)

    # Align the single-trial MUA around the selected event onset
    ntrials = ev_times.size
    channels = [f'MCC{i}' for i in range(1,17)] + [f'LPFC{i}' for i in range(1,17)]
    nch = len(channels)
    w1, w2 = 1000, 1001
    times = np.arange(-w1, w2)
    ntimes = w1 + w2
    mua_aligned = np.zeros((ntrials, nch, ntimes))

    for i,t in enumerate(ev_times):
        mua_aligned[i] = mua[:, t-w1 : t+w2]

    # Z-score across all trials & times for each channel
    zmean = mua_aligned.mean(axis=(0,2), keepdims=True)
    zstd = mua_aligned.std(axis=(0,2), keepdims=True)
    muaNormalized = (mua_aligned-zmean)/zstd

    # Format in xarray
    mua_xr = xr.DataArray(muaNormalized, dims=['trials','channels','times'], 
                        coords=[ev_codes, channels, times])

    # Add metadata in trials
    mua_xr = mua_xr.assign_coords(best_target=('trials', best_target))
    mua_xr = mua_xr.assign_coords(trial_in_block=('trials', trials_in_blocks))

    if event == 'fb': 
        mua_xr = mua_xr.assign_coords(sel_target=('trials', target_codes))
    elif event == 'mo': 
        mua_xr = mua_xr.assign_coords(feedback=('trials', fb_codes))

    # Save the xarray
    f1 = f'{session}-MUAe_{event}.nc'
    utils.save_nc(session, f1, mua_xr)
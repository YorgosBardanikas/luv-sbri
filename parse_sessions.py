"""Script that includes a list of all relevant task event codes
and the preprocessing function that parses a session.
"""

import os
import numpy as np
from scipy.io import loadmat
from frites.io import logger
from utils import path_bhv

# ------------------------------------------
#           Task event codes
# ------------------------------------------
BLOCK = 7 # block initiation

BEST_TG1 = 51 # best target on the left
BEST_TG2 = 52 # best target in the middle
BEST_TG3 = 53 # best target on the right
BEST_TG  = [BEST_TG1, BEST_TG2, BEST_TG3]

MVT_ON = 64 # movement onset
REW = 65    # rewarded trial
NREW = 66   # not rewarded trial

TG1 = 121  # selected target on the left
TG2 = 122  # selected target in the middle
TG3 = 123  # selected target on the right
TG_VALID = 125  # target touch validation
TG  = [TG1, TG2, TG3]

NO_TOUCH = 252  # no touch code
# ------------------------------------------

# Handle string for unpacking behavioral content
BHV_STR_TYPICAL = 'behav'
BHV_STR_EXCEPTION = {'po210422_eopd_5384002': 'behav_cont2'}
# ------------------------------------------


# ----------------------
# Preprocessing function
# ----------------------
def parse_session(session):
    """Main preprocessing function that parses a session.
       It loads task event information and behavioral variables.
       It returns a dictionary with all relevant session variables."""
    
    # Load the behavioral data    
    bhv_filename = f'{session}.mat'
    bhv = loadmat(os.path.join(path_bhv, bhv_filename))
    logger.info(f'Session: {session} is loaded.')

    bhv_str = BHV_STR_EXCEPTION.get(session, BHV_STR_TYPICAL)
    event_codes = bhv[bhv_str][:,0]
    event_times = bhv[bhv_str][:,1]*1000

    # Feedback codes
    fb = np.where((event_codes==REW) | (event_codes==NREW))[0]
    fb_codes = event_codes[fb].astype(int)
    fb_times = event_times[fb].astype(int)

    # Movement onset codes
    # If there is no_touch, remove the movement onset of this trial
    mo = np.where(event_codes==MVT_ON)[0]
    no_touch = np.where(event_codes==NO_TOUCH)[0]

    # Find the relevant movement onset that leads to no_touch.
    # It should be the closest one to each no_touch code,
    # so their absolute difference should be the minimum.
    # Save one mo index for each no_touch in a list.
    bad_mo_inds = [np.argmin(np.abs(mo - n_touch)) for n_touch in no_touch]

    # Remove the movement onset indices of the no_touch trials
    mo_corrected = np.delete(mo, bad_mo_inds)
    mo_times = event_times[mo_corrected].astype(int)

    # Selected target codes
    tg = np.where((event_codes==TG1) | (event_codes==TG2) | (event_codes==TG3))[0]
    target_codes_all = event_codes[tg].astype(int)

    # Check whether target selection is followed by target validation
    tg_correct = np.where(event_codes[tg+1].astype(int)==TG_VALID)[0]
    target_codes = target_codes_all[tg_correct]

    # Block codes
    block_ids = np.where(event_codes==BLOCK)[0]
    nblocks = block_ids.size
    block_starts = event_times[block_ids].astype(int)
    session_end = event_times[-1]
    block_ends = np.append(block_starts[1:], session_end)

    # Find the best target for each block
    best_tg_inblock = np.zeros(nblocks)
    for i in range(nblocks):

        start = block_ids[i]
        if i < nblocks-1: end = block_ids[i+1]
        else: end = len(event_codes)

        events_block = event_codes[start:end]
        counts = np.array([np.sum(events_block == btg) for btg in BEST_TG])
        idx_best = np.argmax(counts) # can be 0,1 or 2 for each of the three targets
        best_tg_inblock[i] = TG[idx_best]
    
    # Create metadata for best target and trials in blocks
    trials_in_blocks = []
    best_target = np.zeros_like(fb_times)
    for block_start, block_end, block_id in zip(block_starts, block_ends, best_tg_inblock):

        # Mask the trials of the current block
        in_block = (fb_times > block_start) & (fb_times < block_end)
        # Set the best target
        best_target[in_block] = block_id
        # Find number of trials in the current block
        tr_in_block_size = np.where(in_block)[0].size
        # Create a vector (0,1,2,...) with trials in the current block
        trials_in_blocks.extend(range(tr_in_block_size))

    trials_in_blocks = np.array(trials_in_blocks)

    if target_codes.size != best_target.size: 
        raise ValueError(f'Unequal sizes: {target_codes.size}, {best_target.size}')

    session_dict = {'fb_times': fb_times,
                    'fb_codes': fb_codes,
                    'mo_times': mo_times,
                    'target_codes': target_codes,
                    'best_target': best_target,
                    'trials_in_blocks': trials_in_blocks}

    return session_dict
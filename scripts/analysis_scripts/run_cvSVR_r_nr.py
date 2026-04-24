"""
SVM decoding of information gain from MUA activity.

Trains a cross-validated SVR on a set of only rewarded 
(or unrewarded) trials per session and area,
and tests on the rest of rewarded (or unrewarded) trials.

A different model is constructed across different epochs 
of the learning curve, in a sliding window.
"""

import os
import utils
import pickle
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from frites.io import logger
from joblib import Parallel, delayed
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from run_SVR_early_late import _fit_svr_perms


# Parameter setting
event = 'fb'
ext = 'nrew'
areas = ['mcc','lpfc']
nperms = 128
decim = 10 # decimation factor
bl_end = 40 # block ends at 40th trial
t_windows = {'fb':[0,2000], 'mo':[-200,0]}
codes = {'rew':65, 'nrew':66}

# Create trial epochs with sliding windows
windows = [(start, start+19) for start in range(0, 21, 5)]

filename = 'Q-regressors_sessions_blocks_all.pkl'
with open(os.path.join(utils.bhv_dir, filename),'rb') as handle:
    Q_blocks = pickle.load(handle)
q_regressors = Q_blocks['model']
session_group = Q_blocks['session_names']

for s,session in enumerate(session_group):
    if s in [22,40]: continue # sessions 22,40 doesnt work

    # Load the regressors of the session
    q_regressors_session = q_regressors[s]
    ig = [q['bayes_surprise'] for q in q_regressors_session]
    ig_flat = np.array([r2 for r1 in ig for r2 in r1])

    try: mua = utils.load_mua_data(session, event)
    except FileNotFoundError: continue
    logger.info(f'Session {session} is loaded.')

    # Downsample for computational efficiency
    mua_ = xr.apply_ufunc(savgol_filter, mua, kwargs={'window_length': 51,'polyorder':2})
    mua = mua_.isel(times=np.arange(0, mua.times.size, decim))
    outcomes = mua.trials.values

    # Select only rewarded (or non-rewarded) trials
    code = codes[ext]
    mua_code = mua.sel(trials=code)
    ig_flat = ig_flat[outcomes==code]
    trials_in_blocks = mua_code.trial_in_block

    # Save the trial indices for each epoch
    groups = []
    for start, end in windows:
        mask = (trials_in_blocks >= start) & (trials_in_blocks <= min(end, bl_end))
        indices = np.where(mask)[0]
        groups.append(indices)

    mua_both = utils.get_mua_per_area(mua_code)

    # Initialize output arrays
    svr_both, svr_both_perm = {},{}
    predictions, predictions_shuffled = [],[]
    true_trials = []

    for area in areas:

        t1,t2 = t_windows[event]
        mua_area = mua_both[area].sel(times=slice(t1,t2)).mean('times').values
        
        # Initialize intermediate arrays
        svr_list, svr_models_perm_list = [],[]
        pred, pred_shuffled = [],[]

        for group in groups:

            train, test = train_test_split(group, test_size=0.5, random_state=10)

            # Fit the real SVR
            svr_train = mua_area[train]
            classes_train = ig_flat[train]
            svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
            svr.fit(svr_train, classes_train)
            svr_list.append(svr)

            # Fit the permuted SVRs
            svr_models_perm = Parallel(n_jobs=-1, backend="threading")(
                                            delayed(_fit_svr_perms)
                                            (svr_train, classes_train, seed) 
                                            for seed in range(nperms))
            svr_models_perm_list.append(svr_models_perm)

            # Predict class in the real SVR
            svr_test = mua_area[test]
            pred.append(svr.predict(svr_test)) # (ngroups, ntrials)
            if area == 'mcc': true_trials.append(ig_flat[test])

            # Predict class in the permuted SVRs
            preds_shuff = [pm.predict(svr_test) for pm in svr_models_perm]
            svr_preds_shuff = np.stack(preds_shuff, axis=0) # (nperms, ntrials)
            pred_shuffled.append(svr_preds_shuff) # (ngroups, nperms, ntrials)

        # Append in the output arrays
        predictions.append(pred) # (nareas, ngroups, ntrials)
        predictions_shuffled.append(pred_shuffled) # (nareas, ngroups, nperms, ntrials)
        svr_both[area] = svr_list
        svr_both_perm[area] = svr_models_perm_list

    # Save
    arrays = [svr_both, svr_both_perm, predictions,
              predictions_shuffled, true_trials]
    filenames = [f'{session}-SVR_model_{event}_{ext}.pkl', 
              f'{session}-SVR_model_shuff_{event}_{ext}.pkl',
              f'{session}-SVR_predictions_{event}_{ext}.pkl',
              f'{session}-SVR_predictions_shuffled_{event}_{ext}.pkl',
              f'{session}-SVR_true_trials_{event}_{ext}.pkl']

    for filename, array in zip(filenames, arrays):
        utils.save_pkl(session, filename, array)
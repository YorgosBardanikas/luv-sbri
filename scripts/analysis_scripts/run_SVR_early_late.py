"""
SVM decoding of information gain from MUA activity.
Trains a SVR on early/late trials per session and area,
and tests on late/early trials (includes test on early/late 
for control).
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
from sklearn.linear_model import LinearRegression

# --------------------
# Helper SVR functions
# --------------------

def _fit_svr_perms(data, classes, seed):
    """Fit one permuted SVR model (shuffled class labels)."""
    rng = np.random.default_rng(seed=seed)
    shuffled_classes = rng.permutation(classes)
    svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
    svr.fit(data, shuffled_classes)
    return svr

def _predict_class(svr, data):
    """
    Apply svr.predict across all trials and timepoints.
    data: (ntrials, nchannels, ntimes)
    Returns predictions: (ntrials, ntimes)
    """
    ntr, nch, nt = data.shape
    data_reshaped = data.transpose(0, 2, 1).reshape((ntr*nt, nch))
    return svr.predict(data_reshaped).reshape((ntr, nt))


if __name__ == '__main__':

    # Parameter setting
    event = 'fb'
    ext = 'late'
    areas = ['mcc','lpfc']
    nperms = 128
    decim = 10 # decimation factor
    windows = {'fb':[0,2000], 'mo':[-200,0]}

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

        times = mua.times.values
        outcomes = mua.trials.values
        trials_in_blocks = mua.trial_in_block
        if ext == 'early': mask = trials_in_blocks < 10
        elif ext == 'late': mask = trials_in_blocks > 10

        mua_both = utils.get_mua_per_area(mua)
        mua_sel = {area: v.sel(trials=mask) for area,v in mua_both.items()}

        # Regress-out the effect of outcomes that exist in the information gain
        # keeping effects from previous trials
        resid_model = LinearRegression().fit(outcomes[:,None], ig_flat)
        ig_resid = ig_flat - resid_model.predict(outcomes[:,None])

        # Initialize output arrays
        svr_both, svr_both_perm = {},{}
        predictions, predictions_shuffled = [],[]

        for area in areas:

            t1,t2 = windows[event]
            svr_train = mua_sel[area].sel(times=slice(t1,t2)).mean('times').values

            # Fit the real SVR
            svr = SVR(kernel='linear', C=1.0, epsilon=0.1)
            svr.fit(svr_train, ig_resid[mask])
            svr_both[area] = svr

            # Fit the permuted SVRs
            svr_models_perm = Parallel(n_jobs=-1, backend="threading")(
                                            delayed(_fit_svr_perms)
                                            (svr_train, ig_resid[mask], seed) 
                                            for seed in range(nperms))
            svr_both_perm[area] = svr_models_perm

            # Predict class in the real SVR
            mua_test = mua_both[area].values
            predictions.append(_predict_class(svr, mua_test)) # (nareas, ntrials, ntimes)

            # Predict class in the permuted SVRs
            preds_shuff = [_predict_class(pm, mua_test) for pm in svr_models_perm]
            svr_preds_shuff = np.stack(preds_shuff, axis=0) # (nperms, ntrials, ntimes)
            predictions_shuffled.append(svr_preds_shuff) # (nareas, nperms, ntrials, ntimes)


        # Format in xarray
        trial_count = np.arange(ig_resid.size)
        svr_predictions = xr.DataArray(np.array(predictions), 
                                    dims=['areas','trials','times'],
                                    coords=[areas, trial_count, times])
        svr_predictions = svr_predictions.assign_coords(true_trials=('trials',ig_resid))

        data = np.array(predictions_shuffled).astype(np.float32)
        svr_predictions_shuffled = xr.DataArray(data, 
                                    dims=['areas','perms','trials','times'],
                                    coords=[areas, np.arange(nperms), trial_count, times])   

        # Save
        f1 = f'{session}-SVR_resid_model_{event}_{ext}.pkl'
        utils.save_pkl(session, f1, svr_both)
        f2 = f'{session}-SVR_resid_model_shuff_{event}_{ext}.pkl'
        utils.save_pkl(session, f2, svr_both_perm)
        f3 = f'{session}-SVR_resid_predictions_{event}_{ext}.nc'
        utils.save_nc(session, f3, svr_predictions)
        f4 = f'{session}-SVR_resid_predictions_shuffled_{event}_{ext}.nc'
        utils.save_nc(session, f4, svr_predictions_shuffled)
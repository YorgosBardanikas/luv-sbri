"""
SVM decoding of trial outcome (reward or no-reward) from MUA activity.
Trains an SVC on early/late trials per session and area,
and tests on late/early trials (includes test on early/late 
for control).
"""

import utils
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from frites.io import logger
from joblib import Parallel, delayed
from sklearn.svm import SVC

# --------------------
# Helper SVC functions
# --------------------

def _fit_svc_perms(data, classes, seed):
    """Fit one permuted SVC model (shuffled class labels)."""
    rng = np.random.default_rng(seed=seed)
    shuffled_classes = rng.permutation(classes)
    svc = SVC(kernel='linear', C=1.0)
    svc.fit(data, shuffled_classes)
    return svc

def _predict_class(svc, data):
    """
    Apply svc.predict across all trials and timepoints.
    data: (ntrials, nchannels, ntimes)
    Returns predictions: (ntrials, ntimes)
    """
    ntr, nch, nt = data.shape
    data_reshaped = data.transpose(0, 2, 1).reshape((ntr*nt, nch))
    return svc.predict(data_reshaped).reshape((ntr, nt))


if __name__ == '__main__':

    # Parameter setting
    event = 'fb'
    ext = 'late'
    areas = ['mcc','lpfc']
    codes = [121,122,123] if event == 'mo' else [65,66]
    nperms = 128
    decim = 10 # decimation factor
    windows = {'fb':[0,2000], 'mo':[-200,0]}

    session_group = utils.get_session_names()

    for session in session_group:

        try: mua = utils.load_mua_data(session, event)
        except FileNotFoundError: continue
        logger.info(f'Session {session} is loaded.')

        # Downsample for computational efficiency
        mua_ = xr.apply_ufunc(savgol_filter, mua, 
                              kwargs={'window_length': 51,'polyorder':2})
        mua = mua_.isel(times=np.arange(0, mua.times.size, decim))

        times = mua.times.values
        regressor = mua.trials.values
        trials_in_blocks = mua.trial_in_block
        if ext == 'early': mask = trials_in_blocks < 10
        elif ext == 'late': mask = trials_in_blocks > 10

        mua_both = utils.get_mua_per_area(mua)
        mua_sel = {area: v.sel(trials=mask) for area,v in mua_both.items()}

        # Initialize output arrays
        svc_both, svc_both_perm = {},{}
        predictions, predictions_shuffled = [],[]

        for area in areas:

            t1,t2 = windows[event]
            mua_train = mua_sel[area].sel(times=slice(t1,t2)).mean('times')

            # Balance the classes
            rgrs = mua_train.trials
            rgr = [np.where(rgrs==n)[0] for n in codes]
            minsize = min(len(r) for r in rgr)
            rng = np.random.default_rng(seed=10)
            inds = np.concatenate([rng.choice(r, size=minsize, replace=False) for r in rgr])

            svc_train = mua_train.isel(trials=inds)
            classes = svc_train.trials.values

            # Fit the real SVC
            svc = SVC(kernel='linear', C=1.0)
            svc.fit(svc_train.values, classes)
            svc_both[area] = svc

            # Fit the permuted SVCs
            svc_models_perm = Parallel(n_jobs=-1, backend="threading")(
                                            delayed(_fit_svc_perms)
                                            (svc_train, classes, seed) 
                                            for seed in range(nperms))
            svc_both_perm[area] = svc_models_perm

            # Predict class in the real SVC (both early and late trials)
            mua_test = mua_both[area].values
            predictions.append(_predict_class(svc, mua_test))

            # Predict class in the permuted SVCs
            preds_shuff = [_predict_class(pm, mua_test) for pm in svc_models_perm]
            svc_preds_shuff = np.stack(preds_shuff, axis=0) # (nperms, ntrials, ntimes)
            predictions_shuffled.append(svc_preds_shuff)


        # Format in xarray
        trial_count = np.arange(regressor.size)
        svc_predictions = xr.DataArray(np.array(predictions), 
                                    dims=['areas','trials','times'],
                                    coords=[areas, trial_count, times])
        svc_predictions = svc_predictions.assign_coords(true_trials=('trials',regressor))

        data = np.array(predictions_shuffled).astype(np.float32)
        svc_predictions_shuffled = xr.DataArray(data, 
                                    dims=['areas','perms','trials','times'],
                                    coords=[areas, np.arange(nperms), trial_count, times])   

        # Save
        f1 = f'{session}-SVC_model_{event}_{ext}.pkl'
        utils.save_pkl(session, f1, svc_both)
        f2 = f'{session}-SVC_model_shuff_{event}_{ext}.pkl'
        utils.save_pkl(session, f2, svc_both_perm)
        f3 = f'{session}-SVC_predictions_{event}_{ext}.nc'
        utils.save_nc(session, f3, svc_predictions)
        f4 = f'{session}-SVC_predictions_shuffled_{event}_{ext}.nc'
        utils.save_nc(session, f4, svc_predictions_shuffled)
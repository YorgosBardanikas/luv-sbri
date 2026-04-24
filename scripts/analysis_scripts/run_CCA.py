"""Apply Canonical Correlation Analysis between MCC and LPFC.
This tests the existence of a shared dimension capturing their 
covariance."""

import utils
import numpy as np
import xarray as xr
from frites.io import logger
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA

# --------------------
# Helper CCA functions
# --------------------

def _compute_cca(x, y, idx_train, idx_test):
    """Apply CCA between MCC and LPFC training and 
    testing on different trials. Returns projections 
    on the CCA axes, shape (ntrials, ncomps)."""

    cca = CCA(n_components=1) # find only top dimension
    cca.fit(x[idx_train], y[idx_train])
    x_cca, y_cca = cca.transform(x[idx_test], y[idx_test])
    return x_cca, y_cca

def _cca_perms(x, y, idx_train, idx_test, seed):
    """Apply CCA after shuffling one of the two arguments.
    Returns projections on random axes, shape (ntrials, ncomps)."""

    rng = np.random.default_rng(seed=seed)
    x_shuf = x[rng.permutation(x.shape[0])]
    cca = CCA(n_components=1)
    cca.fit(x_shuf[idx_train], y[idx_train])
    x_perms, y_perms = cca.transform(x_shuf[idx_test], y[idx_test])
    return x_perms, y_perms


if __name__ == '__main__':

    # Parameter setting
    event = 'fb'
    areas = ['mcc','lpfc']
    decim = 10 # decimation factor
    nperms = 192

    session_group = utils.get_session_names()

    for session in session_group:

        try: mua = utils.load_mua_data(session, event)
        except FileNotFoundError: continue
        logger.info(f'Session {session} is loaded.')

        # Downsample for computational efficiency
        mua_ = xr.apply_ufunc(savgol_filter, mua, 
                                kwargs={'window_length': 51,'polyorder':2})
        mua = mua_.isel(times=np.arange(0, mua.times.size, decim))
        
        trials = mua.trials.data
        ntrials = trials.size
        times = mua.times.data
        ntimes = times.size
        trials_in_blocks = mua.trial_in_block.data

        # Split trials in train and test
        indices = np.arange(ntrials)
        idx_train, idx_test = train_test_split(indices, test_size=0.5, random_state=10)
        ntest = idx_test.size

        # Keep only highly modulated channels, for ensuring CCA convergence
        mua_both = utils.get_mua_per_area(mua)

        mua_new = {}
        for name, mua_area in mua_both.items():
            chans = mua_area.max(2).mean(0) # max over times, mean over trials
            prc = np.percentile(chans, q=50)
            mua_new[name] = mua_area[:, chans > prc, :]

        # Initialize output arrays
        cca_mcc = np.zeros((ntest, ntimes))
        cca_lpfc = np.zeros_like(cca_mcc)
        perms_mcc = np.zeros((nperms, ntest, ntimes))
        perms_lpfc = np.zeros_like(perms_mcc)
        
        for t in range(ntimes):
            if t % 10 == 0: logger.info(f' {session} | time {t}/{ntimes}')

            mcc_t = mua_new['mcc'][...,t]
            lpfc_t = mua_new['lpfc'][...,t]

            # Apply real CCA
            x_cca, y_cca = _compute_cca(mcc_t, lpfc_t, idx_train, idx_test)
            cca_mcc[:,t] = x_cca.squeeze()
            cca_lpfc[:,t] = y_cca.squeeze()

            # Apply shuffled CCA
            results = Parallel(n_jobs=-1)(delayed(_cca_perms)
                                                (mcc_t, lpfc_t, idx_train, idx_test, seed)
                                                for seed in range(nperms))
            
            x_perms = np.array([r[0].squeeze() for r in results])  # (nperms, ntest)
            y_perms = np.array([r[1].squeeze() for r in results])
            perms_mcc[:,:,t] = x_perms
            perms_lpfc[:,:,t] = y_perms


        # Format in xarray
        trial_count = np.arange(idx_test.size)
        trials_in_blocks_ = trials_in_blocks[idx_test]
        perms = np.arange(nperms)

        cca_dimensions = np.stack([cca_mcc, cca_lpfc], axis=0)
        cca_dimensions = xr.DataArray(cca_dimensions, dims=['areas','trials','times'],
                                                coords=[areas, trial_count, times])
        cca_dimensions = cca_dimensions.assign_coords(trial_in_block=('trials',trials_in_blocks_))

        cca_perms = np.stack([perms_mcc, perms_lpfc], axis=0)
        cca_perms = xr.DataArray(cca_perms, dims=['areas','perms','trials','times'],
                                    coords=[areas, perms, trial_count, times])
        
        # Save
        f1 = f'{session}_CCA_dimensions_{event}.nc'
        utils.save_nc(session, f1, cca_dimensions)
        f2 = f'{session}_CCA_perms_{event}.nc'
        utils.save_nc(session, f2, cca_perms)
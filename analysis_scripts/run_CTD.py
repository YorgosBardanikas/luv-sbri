"""
Cross-temporal decoding of direction across trials. 
Train on early trials and test on late trials.
Tests whether the same rules can be used to classify 
directions across learning (stable vs dynamic representations).
"""

import os
import utils
import numpy as np
import xarray as xr
from frites.io import logger
from utils import fig_dir
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ---------------------
# Helper CTD functions
# ---------------------

def _train_decoder(mua_early, y_early):
    """Train a LDA decoder at each early-time point."""

    ntimes = mua_early.times.size
    lda_models = []
    for t in range(ntimes):
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        lda.fit(mua_early[:,:,t], y_early)
        lda_models.append(lda)
    return lda_models


def _test_decoder(mua, y, lda_models):
    """Test the decoder for each early-time point on 
    late-time points (or early for control)."""

    ntimes = mua.times.size
    acc_matrix = np.zeros((ntimes, ntimes))

    for t_train, lda in enumerate(lda_models):
        preds = np.array([lda.predict(mua[:,:,t]) for t in range(ntimes)])
        # count correct predictions over incorrect (average over trials)
        acc_matrix[t_train] = (preds == y[:,None]).mean(axis=0) 

    return acc_matrix


def _perm_decoder(mua_early, y_early, mua_late, y_late, seed):
    """Train a decoder on shuffled classes of the early trials
    and test it on late trials."""    

    logger.info(f'Permutation: {seed+1}')
    rng_perm = np.random.default_rng(seed)
    # shuffle the y of early trials
    y_early_perm = rng_perm.permutation(y_early)
    lda_perm_models = _train_decoder(mua_early, y_early_perm)
    acc_perm = _test_decoder(mua_late, y_late, lda_perm_models)

    return acc_perm


if __name__ == '__main__':

    # Parameter setting
    event = 'mo' # movement onset
    areas = ['mcc','lpfc']
    phases = [100,101] # early, late
    decim = 10 # decimation factor
    nperms = 128

    session_group = utils.get_session_names()

    for session in session_group:

        try: mua = utils.load_mua_data(session, event)
        except FileNotFoundError: continue
        logger.info(f'Session {session} is loaded.')

        # Downsample for computational efficiency
        mua_ = xr.apply_ufunc(savgol_filter, mua, 
                              kwargs={'window_length': 51,'polyorder':2})
        mua = mua_.isel(times=np.arange(0, mua.times.size, decim))

        times = mua.times.data
        directions = mua.trials.data
        trials_in_blocks = mua.trial_in_block.data

        # Map trial blocks to phase codes: early 100, late 101
        phase_map = np.full_like(trials_in_blocks, -1)
        phase_map[trials_in_blocks < 10] = phases[0]
        phase_map[(trials_in_blocks > 20) & (trials_in_blocks < 30)] = phases[1]
        valid = phase_map != -1

        directions = directions[valid]
        trials_in_blocks = phase_map[valid]
        
        mua_both = utils.get_mua_per_area(mua)

        for a,area in enumerate(areas):

            mua_area = mua_both[area][valid]

            mua_early = mua_area[trials_in_blocks==phases[0]] # early
            y_early = directions[trials_in_blocks==phases[0]]
            mua_late = mua_area[trials_in_blocks==phases[1]] # late
            y_late = directions[trials_in_blocks==phases[1]]

            lda_models = _train_decoder(mua_early, y_early)
            acc_matrix = _test_decoder(mua_late, y_late)

            perm_matrices = np.array(Parallel(n_jobs=-1)(delayed(_perm_decoder)
                                    (mua_early, y_early, mua_late, y_late, seed)
                                                    for seed in range(nperms)))

        # Plot accuracies above chance
        plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,4))
        for a, (acc, perms) in enumerate(zip(acc_matrix, perm_matrices)):

            prcnt = np.percentile(perms, 97.5)
            signif = np.where(acc > prcnt, acc, np.nan)

            plt.subplot(1,2,a+1)
            plt.pcolormesh(times, times, signif, vmin=0.3, vmax=0.6)
            plt.axhline(0, color='k', linestyle='dashed', lw=1)
            plt.axvline(0, color='k', linestyle='dashed', lw=1)
        plt.colorbar()

        # Save the figure
        fig_name = f'{session}_CTG_mo_early-late.png'
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=100), plt.close()

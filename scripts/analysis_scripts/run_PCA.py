"""
Principal Component Analysis across channels per area.
It reduces the dimensionality (channels) of the MUA.
"""

import utils
import numpy as np
import xarray as xr
from frites.io import logger
from sklearn.decomposition import PCA

session_group = utils.get_session_names()
event = 'fb'
t1,t2 = 0,2000
ncomps = 1 # select top component (explaining most variance)

for session in session_group:

    try: mua = utils.load_mua_data(session, event)
    except FileNotFoundError: continue
    logger.info(f'Session {session} is loaded.')

    # Unpack the data
    trials = mua.trials.data
    ntrials = trials.size
    times = mua.times.data
    ntimes = times.size
    nchannels = mua.channels.size//2 # /2 because nchannels is per area
    target_codes = mua.sel_target.data
    best_target = mua.best_target.data
    trials_in_blocks = mua.trial_in_block.data
    mua_both = utils.get_mua_per_area(mua)

    # Initialize output arrays
    scores = np.zeros((2,ntrials,ntimes,ncomps))
    comps, expvar = np.zeros(2),np.zeros(2)

    for i, (name, mua_area) in enumerate(mua_both.items()):

        # Fit the PCA
        mua_area_fit = mua_area.sel(times=slice(t1,t2)).mean('times')
        pca = PCA(n_components=ncomps)
        pca.fit(mua_area_fit)
        comps[i] = pca.components_
        expvar[i] = np.round(pca.explained_variance_ratio_*100,1)
        logger.info(f'{name}: {expvar[i]}%')

        # Project the MUA on the PCs
        muaT = mua_area.transpose('trials','times','channels').data
        mua_transform = muaT.reshape((ntrials*ntimes,nchannels))
        scores_area = pca.transform(mua_transform)
        scores[i] = scores_area.reshape((ntrials,ntimes,ncomps))

    # Format in xarray
    areas = list(mua_both.keys())
    pcs = [f'PC{i+1}' for i in range(ncomps)]
    coord_dict = {'areas'         : areas,
                  'trials'        : trials,
                  'times'         : times,
                  'pcs'           : pcs,
                  'sel_target'    : ('trials', target_codes),
                  'best_target'   : ('trials', best_target),
                  'trial_in_block': ('trials', trials_in_blocks),
                  'exp_var'       : ('areas', expvar),
                  'components'    : ('pcs', comps)}

    pca_scores = xr.DataArray(scores, dims=['areas','trials','times','pcs'], 
                                      coords=coord_dict)

    # Save
    xr_filename = f'{session}-PCA_{event}.nc'
    utils.save_nc(session, xr_filename, pca_scores)
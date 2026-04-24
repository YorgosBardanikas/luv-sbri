"""
Representational Distance Matrix across directions for 
early and late phases of learning and each area separately.
Tests whether directional representation is stable or dynamic
across learning.
"""

import os
import utils
import numpy as np
import xarray as xr
from frites.io import logger
from utils import fig_dir
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# -------------------
# Computing functions
# -------------------

def _get_rdm(data, phase, dirs, directions, trials_in_blocks):
    """Build representational distance matrix. 
    Uses correlation as metric (can be euclidean distance).
    Returns shape (npairs). For 3 directions (1-2, 1-3, 2-3)."""
    cond_means = []

    for d in dirs:
        mask = (trials_in_blocks == phase) & (directions == d)
        cond_means.append(data[mask].mean(0))
    cond_means = np.stack(cond_means)

    # Compute and build the square RDM
    rdm = squareform(pdist(cond_means, 
                           metric=lambda u,v: 1-np.corrcoef(u,v)[0,1]))
    # Get the indices of the upper triangle and flatten the rdm
    rdm_flat = rdm[np.triu_indices_from(rdm, k=1)]

    return rdm_flat


def _corrcoef(A, B, axis=1):
    """Compute correlation between A and B (across second axis).
    A,B shapes: (nareas,ndirs,ntimes). Returns corr of shape (nareas,ntimes)."""
    A  = A - A.mean(axis, keepdims=True)
    B  = B - B.mean(axis, keepdims=True)
    num   = (A*B).sum(axis)
    denom = np.sqrt((A**2).sum(axis) * (B**2).sum(axis))
    corr = num / denom
    return corr


if __name__ == '__main__':
    
    # Parameter setting
    event = 'mo' # movement onset
    areas = ['mcc','lpfc']
    clrs = ['navy','firebrick']
    dirs = [121,122,123]
    phases = [100,101] # early, late
    decim = 10 # decimation factor

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
        ntimes = times.size
        npairs = len(dirs) * (len(dirs) - 1) // 2   
        rdm = np.zeros((len(areas),len(phases),npairs,ntimes))

        # Get the representational distance of the directions across phases 
        for a,area in enumerate(areas):
            mua_area = mua_both[area][valid]

            for c,code in enumerate(phases):
                for t in range(ntimes):
                    rdm[a,c,:,t] = _get_rdm(mua_area[...,t], code, dirs,
                                             directions, trials_in_blocks)

        axis = 1 # rdm[:,0,...] has shape (areas,dirs,times), correlation across dirs
        similarity = _corrcoef(rdm[:,0,...], rdm[:,1,...], axis=axis)  # (areas,times)

        # Plot
        plt.figure()
        for i,sim in similarity:
            plt.plot(times, abs(sim), color=clrs[i])
            plt.axvline(0, color='k', linestyle='dashed', lw=1)
            plt.gca().spines[['right','top']].set_visible(False)
        
        # Save the figure
        fig_name = f'{session}_RDM_mo.png'
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=100), plt.close()
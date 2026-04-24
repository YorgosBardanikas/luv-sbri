"""Script to compute the variance of the Information Gain (IG),
regress-out the outcome predictions from the IG and compute
the residual variance as well as IG-outcome correlation."""

import os
import pickle
import utils
import numpy as np
from frites.io import logger
from utils import path_bhv
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Parameter setting
rgr_name = 'bayes_surprise'
event = 'fb'
nshuffles = 200

# Load the regressors
filename = 'Q-regressors_sessions_blocks_all.pkl'
with open(os.path.join(path_bhv,filename),'rb') as handle:
    Q_blocks = pickle.load(handle)
session_group = Q_blocks['session_names']
q_regressors = Q_blocks['model']

l1,l2,l3,l4 = [],[],[],[]
sessions, regressors_correlations, correlations_shuffled = [],[],[]

for s,session in enumerate(session_group): # error in session 40
    if s in [22,40]: continue # sessions 22,40 doesnt work

    # Load the mua
    try: mua = utils.load_mua_data(session, event)
    except FileNotFoundError: continue
    logger.info(f'Session {session} is loaded.')
    outcomes = mua.trials.data
    ntrials = outcomes.size

    # Load the regressors of the session and flatten the (blocks,trials) to (trials,)
    q_regressors_session = q_regressors[s]
    ig = [q[rgr_name] for q in q_regressors_session]
    ig_flat = np.array([r2 for r1 in ig for r2 in r1]) 
    sz = ig_flat.size // 2

    # Split trials in train and test groups
    rng = np.random.default_rng(seed=10)
    train_inds = rng.choice(ig_flat.size, size=sz, replace=False)
    outcomes_train, ig_train = outcomes[train_inds], ig_flat[train_inds]
    outcomes_test, ig_test = outcomes[~train_inds], ig_flat[~train_inds]
    ntrials_test = ig_test.size

    # Fit a linear regression model between the outcomes and information gain
    resid_model = LinearRegression().fit(outcomes_train[:,None], ig_train)

    # Predict outcome values (project them on the IG axis)
    resid_model_predictions = resid_model.predict(outcomes_test[:,None])

    # Regress-out the outcome predictions from the IG values
    ig_resid_all = ig_test - resid_model_predictions

    # Get the correlation between these predictions and the real IG values
    regressors_correlations.append(pearsonr(ig_test, resid_model_predictions)[0])

    # Compute correlations between these predictions and shuffled IG values
    ig_resid_shuffled = np.zeros((nshuffles,ntrials_test))
    corr_shuffled = np.zeros(nshuffles)
    for sh in range(nshuffles):
        rng = np.random.default_rng(seed=sh)
        idx_shuf = rng.permutation(ntrials_test)
        ig_resid_shuffled[sh] = ig_test[idx_shuf] - resid_model_predictions
        corr_shuffled[sh] = pearsonr(ig_test[idx_shuf], resid_model_predictions)[0]
    correlations_shuffled.append(corr_shuffled)
    
    # Compute the variance of IG, residual IG, fraction of variances
    var1 = np.var(ig_test, ddof=1)
    var2 = np.var(ig_resid_all, ddof=1)
    var_shuff = np.var(ig_resid_shuffled, axis=1, ddof=1)
    fraction_of_var = var2 / var1
    fraction_of_var_shuff = var_shuff / var1

    # Save in lists
    l1.append(np.round(var1,3))
    l2.append(np.round(var2,3))
    l3.append(np.round(fraction_of_var,2))
    l4.append(np.round(fraction_of_var_shuff,2))
    sessions.append(session)

output_dict = {'Session_names':sessions,
               'Var_ig':l1,
               'Var_resid':l2,
               'Fraction_resid/ig':l3,
               'Fraction_shuff':l4,
               'correlations':regressors_correlations,
               'correlations_shuffled':correlations_shuffled}

filename = 'Correlation_&_residual_variance.pkl'
with open(os.path.join(path_bhv, filename),'wb') as handle:
    pickle.dump(output_dict, handle)
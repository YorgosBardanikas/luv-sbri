"""Script that fits a Q-learning model to the behavioral data.
It applies a grid-search to find the optimal model parameters.
It saves a result dictionary with the q-regressors, the behavioral
performance and session names that will be used in downstream analyses."""

import os
import pickle
import numpy as np
from frites.io import logger
from scipy.special import softmax
from parse_sessions import parse_session
from utils import path_mua, path_bhv, sessions_to_drop

# --------------------------
# Q-learning model functions
# --------------------------

def qlearning(act, rew, alpha, beta, q0=0.5):
    """Q-learning model.
    
    Parameters
    ----------
    act : array | int
        Sequence of actions {1, ..., nActions}
    rew : array | int
        Sequence of rewards {0,1}
    alpha : float
        Learning rate {0->1}
    beta : float
        Temperature of softmax
    q0 : float
        Initial value of Q

    Returns
    -------
    regs : dict
        LL: Log-likelihood with the data.
        Pcor: Probability to choose correct
        RPE: Reward Prediction Error
        BS: Bayesian Surprise

    """

    # Number of actions and trials
    nA, nT = act.max()+1, act.shape[0]

    # Init vars
    LL = 0.0
    RPE = np.zeros((nT))
    Pcor = np.zeros((nT))
    BS = np.zeros((nT))

    # Init Q-values
    Q = q0 * np.ones(nA)

    for i in range(nT):

        # P(act): action probabilities
        P = softmax(beta*Q, axis=0)
        Pcor[i] = P[act[i]]

        # Update Q-values with reward prediction errors
        rpe = alpha * (rew[i] - Q[act[i]])
        Q[act[i]] += rpe
        RPE[i] = rpe

        # Bayesian Surprise (update of beliefs)
        Pu = softmax(beta*Q, axis=0) # Updated Probabilities
        BS[i] = np.sum(P[act[i]] * np.log(P[act[i]] / Pu[act[i]]))

        # Log-likelihood given the action
        LL += np.log(Pcor[i])

    # Create ditionary with regressors
    regs = {'LL': LL, 'Pcor': Pcor, 'rpe': RPE, 'bayes_surprise': BS}

    return regs


def fit_qlearning(act, rew):
    """Fit Q-learning model to behavioral data

    Parameters
    ----------
    act : array | int
        Sequence of actions {1, ..., nActions}
    rew : array | int
        Sequence of rewards {0,1}

    Returns
    -------
    LL : float
        Log-likelihood with the data

    """

    # Init free params
    LL = float('-Inf')
    # Learning rate
    alphas = np.arange(0.1, 1, 0.01)
    # inverse temperature softmax
    betas = np.arange(1, 20, 1)

    # Grid search
    for a in alphas:
        # logger.info(np.round(a,2))
        for b in betas:

            # Run Q-learning
            regs = qlearning(act, rew, a, b)

            # Store maximum log-likelohood
            if regs['LL'] > LL:
                # Add fields
                alpha_fit, beta_fit = a, b
                regs['alpha_fit'] = np.round(alpha_fit,2)
                regs['beta_fit'] = beta_fit
                # Update Log-Likelihood
                LL = regs['LL']
                # Output best fitting regressors
                regs_fit = regs

    return regs_fit


if __name__ == '__main__':

    session_group = sorted(os.listdir(path_mua))
    performance, q_regressors_sessions, feedbacks_sessions, session_names = [],[],[],[]

    for session in session_group:

        if session in sessions_to_drop: continue
        session_dict = parse_session(session)
        feedback_codes = session_dict['fb_codes']
        trials_in_blocks = session_dict['trials_in_blocks']
        target_codes = session_dict['target_codes']
        best_target = session_dict['best_target']

        # Remap the codes of rewards to 1 = 65 (reward), 0 = 66 (no-reward)
        codes,remap = [65,66],[1,0]
        for code, rm in zip(codes,remap):
            feedback_codes = np.where(feedback_codes==code, rm, feedback_codes) 

        # Find the moments of block change (derivative different than 1)
        d = np.diff(trials_in_blocks)
        d_ = np.concatenate(([-10],d,[-10])) # Add block changes to start and end
        # The block changes when the difference between two consecutive trials is different than 1
        block_changes = np.where(d_ != 1)[0]
        nblocks = len(block_changes) - 1 

        # Arrange the target codes and feedback codes in blocks
        starts, ends = block_changes[:-1], block_changes[1:]
        feedback_blocks = [feedback_codes[start:end] for start,end in zip(starts, ends)]
        tg_codes_q = target_codes - 121 # subtract 121 to make the codes [0,1,2]
        target_blocks = [tg_codes_q[start:end] for start,end in zip(starts, ends)]

        # Compute percentage of best target selected
        validity = target_codes == best_target
        uniq_in_block = np.unique(trials_in_blocks)
        avg_validity = np.zeros(uniq_in_block.size)
        for v,value in enumerate(uniq_in_block):
            avg_validity[v] = validity[trials_in_blocks==value].mean()
        performance.append(avg_validity)

        # Fit Q-learning model to the data (different model per learning block)
        q_regressors_blocks = []
        for b, (targets, feedbacks) in enumerate(zip(target_blocks, feedback_blocks)):
            logger.info(f'Block #{b+1}/{nblocks}')
            q_regressors = fit_qlearning(targets, feedbacks)
            q_regressors_blocks.append(q_regressors)

        # Save session results in lists
        q_regressors_sessions.append(q_regressors_blocks)
        feedbacks_sessions.append(feedback_blocks)
        session_names.append(session)

    # Save the data
    output_dict = {'session_names':session_names, 
                   'data':performance, 
                   'model':q_regressors_sessions, 
                   'feedback':feedbacks_sessions}
    
    filename = 'Q-regressors_sessions_blocks_all.pkl'
    with open(os.path.join(path_bhv, filename),'wb') as handle:
        pickle.dump(output_dict, handle)
    logger.info(f'Q-Regressors are saved in the file: "{filename}".')
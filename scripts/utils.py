"""Paths to data directories and utility functions."""
import os
import pickle
import xarray as xr
from frites.io import logger

# ----------------
# Paths
# ----------------

parent = '/envau/work/comco/bardanikas.g/ProcykLab'
data_dir = '/MUA'
bhv_dir = '/Behavioral'
fig_dir = '/Figures'
path_mua = os.path.join(parent, data_dir)
path_bhv = os.path.join(parent, bhv_dir)

# ----------------
# Sessions to drop
# ----------------

sessions_to_drop = [# mismatch trials between target_sel and best_target
                    'po150920_eopk_5587002','po310822_gjld_5522001','ka310522_fpkk_4785003',
                    'ka270720_fikj_7518002','ka250121_eklh_7844002','ka220221_dfje_5088002',
                    'ka210722_dpmn_7579001','ka200620_fjmg_7167001','ka170720_gnli_9009003',
                    'ka050620_ejjf_6914002','ka010822_eflm_6187002',
                    # bad MUA sessions
                    'ka011020_fhkc_6455003','ka030820_dkqh_8215002','po140121_dnni_4785005',
                    'po180322_ekqe_6067005','po200722_dile_5782002','po210422_eopd_5384002',
                    'po280722_ghpg_5869002',
                    # mismatch trials between mo and fb (no_touch corrected)
                    'po140920_dhnh_5252002']

# ------------------
# Utility functions
# ------------------

def get_session_names():
    filename = 'Q-regressors_sessions_blocks_all.pkl'
    with open(os.path.join(path_bhv, filename),'rb') as handle:
        Q_blocks = pickle.load(handle)
    return Q_blocks['session_names']

def load_mua_data(session, event):
    mua_filename = f'{session}-MUAe_{event}.nc'
    mua = xr.open_dataarray(os.path.join(path_mua, session, mua_filename), engine='h5netcdf')
    return mua

def get_mua_per_area(mua):
    mua_mcc = mua.isel(channels=slice(0, 16))
    mua_lpfc = mua.isel(channels=slice(16, 32))
    return {'mcc': mua_mcc, 'lpfc': mua_lpfc}

def save_pkl(session, filename, obj):
    full = os.path.join(path_mua, session, filename)
    with open(full, 'wb') as handle:
        pickle.dump(obj, handle)
    logger.info(f'Saved: "{full}"')

def save_nc(session, filename, da):
    full = os.path.join(path_mua, session, filename)
    da.to_netcdf(full, engine='h5netcdf')
    logger.info(f'Saved: "{full}"')

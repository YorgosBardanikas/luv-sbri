"""Script to get the coordinates of the implanted 
chamber per session."""

import os
import utils
import pickle
import string
import numpy as np

session_group = utils.get_session_names()
coords = []

for session_name in session_group:

    # Get the coordinates of the recording sites
    letters = list(string.ascii_lowercase[0:19])
    seq = np.arange(19)
    dct = dict(zip(letters,seq))
    coordLetters = session_name[9:13]
    coordNumbers = []
    for cL in coordLetters:
        coordNumbers.append(dct[cL])
    coords.append(coordNumbers)

# Save the output in a dictionary
output_dict = {'coords':coords, 'session_names':session_group}
filename = 'Chamber_coordinates_sessions.pkl'
with open(os.path.join(utils.path_bhv, filename),'wb') as handle:
    pickle.dump(output_dict, handle) 
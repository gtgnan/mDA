# generic import
import numpy as np
import itertools as iter
import os
import shutil

# mne import
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne import set_log_level

set_log_level(verbose=False)

# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import itertools as iter
import scipy as sc
import pickle as pk

# mne import
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne import set_log_level

# pyriemann import
import pyriemann
from pyriemann.classification import MDM, TSclassifier, class_distinctiveness
from pyriemann.estimation import Covariances, Coherences
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from pyriemann.embedding import SpectralEmbedding
from pyriemann.transfer import encode_domains, decode_domains, TLCenter, TLStretch, TLRotate
from pyriemann.utils.mean import mean_covariance
from pyriemann.clustering import Kmeans

# sklearn imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit, BaseCrossValidator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import itertools

set_log_level(verbose=False)

selected_events = None
inconsistent_session = False

from moabb.datasets import Shin2017A

# ====================================
# ============ Shin2017A =============
# ====================================

dataset1 = Shin2017A(accept=True)
subjects = [i+1 for i in range(29)] # 29
sessions = ["0imagery", "2imagery", "4imagery"]
runs = ['0']
tmin, tmax = 1., 5.
sample_step = 1 # chan = 30
dispersion_1 = 50
dispersion_2 = 500
dispersion_3 = 100

# For Shin2017A
# ori:  {'left_hand': 1, 'right_hand': 2}
# set:  {'idle': 1, 'left_hand': 2, 'right_hand': 3}

# ====================================

n_subjects = len(subjects)
n_sessions = len(sessions)
n_runs = len(runs)

# ========================

all_epochs = []
all_labels = []
data_path = os.path.expanduser('~/mne_data/')

for subject in subjects:
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    print(f"Loading subjects {subject}...")
    data = dataset1.get_data(subjects=[subject])

    print(f"Processing raw data for subject {subject}")
    # ========================
    
    raw_files = [
        data[subject][ses][run] for ses, run in iter.product(sessions, runs)
    ]
    raw = concatenate_raws(raw_files)
        
    picks = pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    # subsample elecs
    picks = picks[::sample_step]

    # Apply band-pass filter
    raw.filter(7., 35., method='iir', picks=picks)

    events, event_ids = mne.events_from_annotations(raw, event_id = selected_events)

    # Reannotation
    # Get the annotations
    annotations = raw.annotations.copy()

    # Loop through annotations to find trial onsets
    for idx, annot in enumerate(raw.annotations):  # Iterate until the second to last annotation
        # Create a new annotation for resting-state
        if selected_events:
            if raw.annotations[idx]['description'] not in selected_events:
                continue
        if raw.annotations[idx]['duration'] > 0:
            duration = raw.annotations[idx]['duration']
            duration = max(1, duration)
            onset = annot['onset'] - duration
            description = 'idle'

            # Add this resting-state period as a new annotation
            annotations.append(onset, duration, description)

    # Update the annotations in the raw data
    raw.set_annotations(annotations)

    # Extract events including resting-state periods
    if selected_events:
        events, event_ids = mne.events_from_annotations(raw)
        selected_events['idle'] = event_ids['idle']
    events, event_ids = mne.events_from_annotations(raw, event_id = selected_events)
    
    # ========================

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(
        raw,
        events,
        None,
        tmin,
        tmax,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose=False)
    labels = epochs.events[:, -1]

    print("Shape of events: ", events.shape[0])
    print("Shape of labels: ", labels.shape[0])
    
    all_epochs.append(epochs)
    all_labels.append(labels)

    shutil.rmtree(data_path)

epoch_path = "../epoch_data/Shin2017A"

# Check if the directory already exists
if not os.path.exists(epoch_path):
    # Create the parent directory if it doesn't exist
    os.makedirs(os.path.dirname(epoch_path), exist_ok=True)
    print(f"Directory '{epoch_path}' created successfully.")
else:
    print(f"Directory '{epoch_path}' already exists.")

for i, sub in enumerate(subjects):
    with open(f"{epoch_path}/sch_{sub}_epoch.pkl", 'wb') as f:
        pk.dump(all_epochs[i], f)
    with open(f"{epoch_path}/sch_{sub}_label.pkl", 'wb') as f:
        pk.dump(all_labels[i], f)

print("Save epoched data to epoch_data. Execute mDA_Schirrmeister.ipynb to show the analysis.")

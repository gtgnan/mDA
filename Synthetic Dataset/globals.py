import numpy as np

stim_nonErrp = 1
stim_Errp = 0
freq = 125.0
channels = ['Pz', 'F4', 'C4', 'P4', 'O2', 'F8', 'Fp2', 'Cz', 'Fz', 'F3', 'C3', 'P3', 'O1', 'F7', 'Fp1', 'Fpz']
selected_channels = [0,1,2,3,4,7,8,9,10,11,12] # If empty means all channels are selected, other option: [0,4,5]
xdawn_filters = 5
xdawn_filters_time = 5
xdawn_filters_freq = 5
xdawn_filters_spatial = 5
algo = "ner" # options are mit,ner,custom
classifier_type = "Linear"
clf_subcat = "Not_KNN"
action_group=4
kernel_freq, kernel_time = 'linear', 'linear'
total_channels = 16 if selected_channels is None else len(selected_channels)
if selected_channels:
    channels = [channels[i] for i in selected_channels]

data_dict_1 = {}

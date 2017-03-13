"""Prediction stuff for FlightDelay project."""

import numpy as np

##
##
##

def make_onehot_feat_dict(df, feat_key, feat_name):
    """   """

    # Create features for each day of the week
    feat_vals = df[feat_key].values
    all_vals = np.unique(feat_vals)
    N_vals = len(all_vals)
    N_feat = N_vals - 1

    # Create keys
    keys = [0]*N_feat
    for i in range(N_feat):
        keys[i] = 'f_'+feat_name+'_'+ str(all_vals[i])

    # Create value for each training example in dict
    feat_dict = {}
    for i, k in enumerate(keys):
        this_day = all_vals[i]
        feat_dict[k] = feat_vals == this_day

    return feat_dict


def make_onehot_feat_mat(df, feat_key):
    """   """

    # Extract the feature data of interest
    feat_dat = df[feat_key].values
    n_dat = len(feat_dat)

    # Get unique possible values of feature
    pos_vals = np.unique(feat_dat)
    n_feat = len(pos_vals)

    labs = dict()
    for i, v in enumerate(pos_vals):
        labs[v] = i

    oh_feats = np.zeros(shape=[n_dat, n_feat])

    for i, d in enumerate(feat_dat):

        oh_feats[i, labs[d]] = 1

    return oh_feats

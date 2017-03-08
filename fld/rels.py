"""Explore relations in the data."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from statsmodels.distributions.empirical_distribution import ECDF

##
##
##

def relation_explorer(df, key_predict, key_feature,
                      analysis_fn, analysis_kwargs = {},
                      restriction = None, feature_is_categorical = True, make_plot = True,
                      return_all_x = False, return_N_x = False):
    """
    Compute the relationship between two features in the data

    Parameters
    ----------
    df : pandas DataFrame
        data
    key_predict : str
        column of `df` to predict
    key_feature : str
        column of `df` to use as the predictive feature
    restriction : None or dict of type {key: list}
        if None: there are no restrictions on which data is included
        if dict: restrict `df` to only the rows in which the key of
        `restriction` takes on one of the values in the list
    feature_is_categorical : bool
        if True, the predictive feature is assumed to be categorical in nature
        if False, the predictive feature is assume to be ordinal in nature

    Returns
    -------
    out : varies
        output of the `analysis_fn`
    """

    # Restrict the data, if necessary
    if restriction is not None:
        df = restrict_df(df, restriction)

    # Determine the thing to predict
    y = df[key_predict].values

    # Determine the predictor
    x = df[key_feature].values

    if feature_is_categorical:
        # Split y by x
        y_by_x, all_x, N_x = split_y_by_x(x, y, return_all_x = True, return_N_x = True)

        # Set default histogram params
        out = analysis_fn(y_by_x, all_x = all_x, make_plot = make_plot, **analysis_kwargs)

    else:
        raise NotImplementedError('TODO')

    if return_all_x:
        if return_N_x:
            return out, all_x, N_x
        else:
            return out, all_x
    else:
        if return_N_x:
            return out, N_x
        else:
            return out


def relation_mean(y_by_x, all_x = None, make_plot = True):
    """Calculate the mean of y for each value of x"""

    N_x = len(y_by_x)
    y_by_x_mean = np.zeros(N_x)
    for i in range(N_x):
        y_by_x_mean[i] = np.mean(y_by_x[i])

    if make_plot:
        if all_x is None:
            all_x = np.arange(N_x)
        plt.plot(np.arange(N_x),y_by_x_mean,'k.-')
        plt.xticks(np.arange(N_x),all_x)

    return y_by_x_mean


def relation_quantiles(y_by_x, all_x = None, all_quantiles = None, make_plot = True):
    """Compute and plot the quantiles of y for all values of x"""

    if all_quantiles is None:
        all_quantiles = np.arange(10,100,10)
    N_quantiles = len(all_quantiles)
    N_x = len(y_by_x)

    y_by_x_quantile = np.zeros((N_x,N_quantiles))
    for i in range(N_x):
        y_by_x_quantile[i] = np.percentile(y_by_x[i],all_quantiles)


    if make_plot:
        if all_x is None:
            all_x = np.arange(N_x)

        quantile_cmap = getcmaprgb(N_quantiles, cm.jet)
        for i in range(N_quantiles):
            plt.plot(np.arange(N_x),y_by_x_quantile[:,i],'.-',color=quantile_cmap[i], label=all_quantiles[i])
        plt.xticks(np.arange(N_x),all_x)
        plt.legend()

    return y_by_x_quantile, all_quantiles


def relation_exceed(y_by_x, all_x = None, all_min = None, make_plot = True):
    """Determine the fraction of observations for each x that y exceeds"""

    if all_min is None:
        all_min = np.percentile(np.hstack(y_by_x),np.arange(10,100,10))
    N_min = len(all_min)
    N_x = len(y_by_x)


    y_by_x_min = np.zeros((N_x,N_min))
    for i in range(N_x):
        for j in range(N_min):
            y_by_x_min[i,j] = np.mean(y_by_x[i]>all_min[j])


    if make_plot:
        if all_x is None:
            all_x = np.arange(N_x)

        min_cmap = getcmaprgb(N_min, cm.jet)
        for i in range(N_min):
            plt.plot(np.arange(N_x),y_by_x_min[:,i],'.-',color=min_cmap[i], label=all_min[i])
        plt.xticks(np.arange(N_x),all_x)
        plt.legend()

    return y_by_x_min, all_min


def relation_ecdf(y_by_x, all_x = None, make_plot = True):
    """Compute the cumulative distribution function of y for all values of x"""

    N_x = len(y_by_x)
    y_by_x_cdfy = np.zeros(N_x, dtype=np.ndarray)
    for i in range(N_x):
        ecdf = ECDF(y_by_x[i])
        y_by_x_cdfy[i] = ecdf(y_by_x[i])

    if make_plot:
        if all_x is None:
            all_x = np.arange(N_x)

        x_cmap = getcmaprgb(N_x, cm.jet)
        for i in range(N_x):
            plt.plot(y_by_x[i],y_by_x_cdfy[i],'.', color=x_cmap[i], label=all_x[i])
        plt.legend()

    return y_by_x_cdfy

#####

def getcmaprgb(N, cmap):
    """Get the RGB values of N colors across a colormap"""
    return cmap(np.linspace(0,255,N).astype(int))

def split_y_by_x(x, y, return_all_x = False, return_N_x = False):
    """Split the values of y into separate arrays for each value in x"""

    all_x = np.unique(x)
    N_x = len(all_x)
    y_by_x = np.zeros(N_x, dtype=np.ndarray)
    for i in range(N_x):
        y_by_x[i] = y[x==all_x[i]]

    if return_all_x:
        if return_N_x:
            return y_by_x, all_x, N_x
        else:
            return y_by_x, all_x
    else:
        if return_N_x:
            return y_by_x, N_x
        else:
            return y_by_x

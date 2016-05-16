"""
Provides a temporal matching function
"""

import numpy as np
from scipy.spatial import cKDTree

import pandas as pd


def df_match(reference, *args, **kwds):
    """
    Finds temporal match between the reference pandas.DataFrame (index has to
    be datetime) and n other pandas.DataFrame (index has to be datetime).

    Parameters
    ----------
    reference : pandas.DataFrame or pandas.TimeSeries
        The index of this dataframe will be the reference.
    *args : pandas.DataFrame or pandas.TimeSeries
        The index of this dataframe(s) will be matched.
        If it is a pandas.Series then it has to have a name. Otherwise
        no column name can be assigned to the matched DataFrame.
    window : float
        Fraction of days of the maximum pos./neg. distance allowed, i.e. the
        value of window represents the half-winow size (e.g. window=0.5, will
        search for matches between -12 and +12 hours) (default: None)
    dropna : boolean
        Drop rows containing only NaNs (default: False)
    dropduplicates : boolean
        Drop duplicated temporal matched (default: False)
    asym_window: string, optional
        ``<=`` stands for using a smaller and equal only for the left/smaller side of the window comparison
        ``>=`` stands for using a larger and equal only for the right/larger side of the window comparison
        The default is to use <= and >= for both sides of the search window

    Returns
    -------
    temporal_matched_args : pandas.DataFrame or tuple of pandas.DataFrame
        Dataframe with index from matched reference index
    """
    if "window" in kwds:
        window = kwds['window']
    else:
        window = None

    if "asym_window" in kwds:
        asym_window = kwds['asym_window']
    else:
        asym_window = None

    temporal_matched_args = []
    ref_step = reference.index.values - reference.index.values[0]

    for arg in args:

        if type(arg) in [pd.TimeSeries, pd.Series]:
            arg = pd.DataFrame(arg)
        comp_step = arg.index.values - reference.index.values[0]
        values = np.arange(comp_step.size)
        # setup kdtree which must get 2D input
        try:
            tree = cKDTree(np.atleast_2d(comp_step).T, balanced_tree=False)
        except TypeError:
            # scipy before version 0.16 does not have the balanced_tree kw
            # but is fast in this case also without it
            tree = cKDTree(np.atleast_2d(comp_step).T)

        dist, i = tree.query(np.atleast_2d(ref_step).T)
        matched = values[i]

        distance = np.zeros_like(matched, dtype=np.float)
        distance.fill(np.nan)
        valid_match = np.invert(np.isnan(matched))

        distance[valid_match] = \
            (arg.index.values[np.int32(matched[valid_match])] -
             reference.index.values[valid_match]) / np.timedelta64(1, 'D')

        arg['index'] = arg.index.values
        arg['merge_key'] = np.arange(len(arg))

        arg_matched = pd.DataFrame({'merge_key': matched,
                                    'distance': distance,
                                    'ref_index': reference.index.values})
        arg_matched = arg_matched.merge(arg, on="merge_key", how="left")
        arg_matched.index = arg_matched['ref_index'].values
        arg_matched = arg_matched.sort_index()

        if window is not None:
            if asym_window is None:
                invalid_dist = arg_matched['distance'].abs() > window
            if asym_window == "<=":
                # this means that only distance in the interval [distance[ are
                # taken
                valid_dist = ((arg_matched['distance'] >= 0.0) & (arg_matched['distance'] <= window)) | (
                    (arg_matched['distance'] <= 0.0) & (arg_matched['distance'] > -window))
                invalid_dist = ~valid_dist
            if asym_window == ">=":
                # this means that only distance in the interval ]distance] are
                # taken
                valid_dist = ((arg_matched['distance'] >= 0.0) & (arg_matched['distance'] < window)) | (
                    (arg_matched['distance'] <= 0.0) & (arg_matched['distance'] >= -window))
                invalid_dist = ~valid_dist
            arg_matched.loc[invalid_dist] = np.nan

        if "dropna" in kwds and kwds['dropna']:
            arg_matched = arg_matched.dropna()

        if "dropduplicates" in kwds and kwds['dropduplicates']:
            arg_matched = arg_matched.dropna()
            g = arg_matched.groupby('merge_key')
            min_dists = g.distance.apply(lambda x: x.abs().idxmin())
            arg_matched = arg_matched.ix[min_dists]

        temporal_matched_args.append(
            arg_matched.drop(['merge_key', 'ref_index'], axis=1))

    if len(temporal_matched_args) == 1:
        return temporal_matched_args[0]
    else:
        return tuple(temporal_matched_args)


def matching(reference, *args, **kwargs):
    """
    Finds temporal match between the reference pandas.TimeSeries (index has to
    be datetime) and n other pandas.TimeSeries (index has to be datetime).

    Parameters
    ----------
    reference : pandas.TimeSeries
        The index of this Series will be the reference.
    *args : pandas.TimeSeries
        The index of these Series(s) will be matched.
    window : float
        Fraction of days of the maximum pos./neg. distance allowed, i.e. the
        value of window represents the half-winow size (e.g. window=0.5, will
        search for matches between -12 and +12 hours) (default: None)

    Returns
    -------
    temporal_match : pandas.DataFrame
        containing the index of the reference Series and a column for each of the
        other input Series
    """
    matched_datasets = df_match(reference, *args, dropna=True,
                                dropduplicates=True, **kwargs)

    if type(matched_datasets) != tuple:
        matched_datasets = [matched_datasets]

    matched_data = pd.DataFrame(reference)
    era_file = 0
    for match in matched_datasets:

        if match.keys()[1]=='sm_era':
            era_pos_args = era_file
        else:
            era_file = era_file+1

        if match.keys()[1]=='sm':
            pos = np.where(np.abs(match['distance']) > 0.25)[0]
            for k in np.arange(pos.size):
                time_smos = match.index[pos[k]]
                pos_era = np.where(matched_datasets[era_pos_args].index==time_smos)[0]
                if pos_era.size>0:
                    time_era = matched_datasets[era_pos_args]['index'][pos_era][0]
                    dist_era = matched_datasets[era_pos_args]['distance'][pos_era][0]
                    if match['distance'][pos[k]]>=0:
                        merge_key = args[era_pos_args]['merge_key'][time_era]
                        if dist_era>=0:
                            #diff_values_era = arsgs[era_pos_args]['sm_era'][merge_key] - args[era_pos_args]['sm_era'][merge_key+1]
                            values_era = args[era_pos_args]['sm_era'][merge_key:merge_key+2]
                        else:
                            #diff_values_era = 2*arsgs[era_pos_args]['sm_era'][merge_key] - args[era_pos_args]['sm_era'][merge__key+1] - args[era_pos_args]['sm_era'][merge_key+2]
                            values_era = args[era_pos_args]['sm_era'][merge_key:merge_key+3]
                    else:
                        merge_key = args[era_pos_args]['merge_key'][time_era]
                        if dist_era<0:
                            #diff_values_era = args[era_pos_args]['sm_era'][merge_key]-args[era_pos_args]['sm_era'][merge_key-1]
                            values_era = args[era_pos_args]['sm_era'][merge_key-1:merge_key+1]
                        else:
                            #diff_values_era = 2*args[era_pos_args]['sm_era'][merge_key]-args[era_pos_args]['sm_era'][merge_key-1]-args[era_pos_args]['sm_era'][merge_key-2]
                            values_era = args[era_pos_args]['sm_era'][merge_key-2:merge_key+1]
                    #if abs(diff_values_era) > 2:
                    if np.std(values_era) > 2:
                        match['sm'][pos[k]] = np.nan
                        #print values_era
                else:
                    match['sm'][pos[k]] = np.nan
                    #print 'ooooooo'
                        #values_era = [0]
        match = match.drop(['distance','index'], axis=1)
        #match = match.drop(['sm','index','ssf'], axis=1)
        matched_data = matched_data.join(match)

    return matched_data.dropna()

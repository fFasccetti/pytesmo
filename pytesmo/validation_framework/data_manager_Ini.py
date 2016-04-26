# Copyright (c) 2015, Vienna University of Technology (TU Wien), Department
# of Geodesy and Geoinformation (GEO).
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology, Department
#     of Geodesy and Geoinformation nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools
import warnings

import pandas as pd


class DataManager(object):

    """
    Class to handle the data management.

    Parameters
    ----------
    datasets : dict of dicts
        Keys: string, datasets names
        Values: dict, containing the following fields
            'class': object
                Class containing the method read_ts for reading the data.
            'columns': list
                List of columns which will be used in the validation process.
            'type': string
                'reference' or 'other'.
            'args': list, optional
                Args for reading the data.
            'kwargs': dict, optional
                Kwargs for reading the data
            'grids_compatible': boolean, optional
                If set to True the grid point index is used directly when
                reading other, if False then lon, lat is used and a nearest
                neighbour search is necessary.
            'use_lut': boolean, optional
                If set to True the grid point index (obtained from a
                calculated lut between reference and other) is used when
                reading other, if False then lon, lat is used and a
                nearest neighbour search is necessary.
            'lut_max_dist': float, optional
                Maximum allowed distance in meters for the lut calculation.
    data_prep : object, optional
        Object that provides the methods prep_reference and prep_other
        which take the pandas.Dataframe provided by the read_ts methods (plus
        other_name for prep_other) and do some data preparation on it before
        temporal matching etc. can be used e.g. for special masking or anomaly
        calculations.
    period : list, optional
        Of type [datetime start, datetime end]. If given then the two input
        datasets will be truncated to start <= dates <= end.

    Methods
    -------
    use_lut(other_name)
        Returns lut between reference and other if use_lut for other dataset
        was set to True.
    get_result_names()
        Return results names based on reference and others names.
    read_reference(*args)
        Function to read and prepare the reference dataset.
    read_other(other_name, *args)
        Function to read and prepare the other datasets.
    """

    def __init__(self, datasets, data_prep=None, period=None):
        """
        Initialize parameters.
        """
        self.datasets = datasets

        self.other_name = []
        for dataset in datasets.keys():
            if datasets[dataset]['type'] == 'reference':
                self.reference_name = dataset
            else:
                self.other_name.append(dataset)

        try:
            self.reference_grid = self.datasets[
                self.reference_name]['class'].grid
        except AttributeError:
            self.reference_grid = None

        self.data_prep = data_prep
        self.period = period

    def get_luts(self):
        """
        Returns luts between reference and others if use_lut for other datasets
        was set to True.

        Returns
        -------
        luts : dict
            Keys: other datasets names
            Values: lut between reference and other, or None
        """
        luts = {}
        for other_name in self.other_name:
            if self.datasets[other_name]['use_lut']:
                luts[other_name] = self.reference_grid.calc_lut(
                    self.datasets[other_name]['class'].grid,
                    max_dist=self.datasets[other_name]['lut_max_dist'])
            else:
                luts[other_name] = None

        return luts

    def get_results_names(self):
        """
        Return results names based on reference and others names.

        Returns
        -------
        results_names : list
            Containing all combinations of
            (referenceDataset.column, otherDataset.column)
        """
        results_names = []

        ref_columns = []
        for column in self.datasets[self.reference_name]['columns']:
            ref_columns.append(self.reference_name + '.' + column)

        other_columns = []
        for other in self.other_name:
            for column in self.datasets[other]['columns']:
                other_columns.append(other + '.' + column)

        for comb in itertools.product(ref_columns, other_columns):
            results_names.append(comb)

        return results_names

    def read_reference(self, *args):
        """
        Function to read and prepare the reference dataset.
        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        ref_df : pandas.DataFrame or None
            Reference dataframe.
        """
        reference = self.datasets[self.reference_name]
        args = list(args)
        args.extend(reference['args'])

        try:
            ref_df = reference['class'].read_ts(*args, **reference['kwargs'])
        except IOError:
            warnings.warn("IOError while reading reference {:}".format(args))
            return None
        except RuntimeError as e:
            if e.args[0] == "No such file or directory":
                warnings.warn(
                    "IOError while reading reference {:}".format(args))
                return None
            else:
                raise e

        if len(ref_df) == 0:
            warnings.warn("No data for reference {:}".format(args))
            return None

        if self.data_prep is not None:
            ref_df = self.data_prep.prep_reference(ref_df)

        if len(ref_df) == 0:
            warnings.warn("No data for reference {:}".format(args))
            return None

        if isinstance(ref_df, pd.DataFrame) == False:
            warnings.warn("Data is not a DataFrame {:}".format(args))
            return None

        if self.period is not None:
            ref_df = ref_df[self.period[0]:self.period[1]]

        if len(ref_df) == 0:
            warnings.warn("No data for reference {:}".format(args))
            return None

        else:
            return ref_df

    def read_other(self, other_name, *args):
        """
        Function to read and prepare the other datasets.
        Takes either 1 (gpi) or 2 (lon, lat) arguments.

        Parameters
        ----------
        other_name : string
            Name of the other dataset.
        gpi : int
            Grid point index
        lon : float
            Longitude of point
        lat : float
            Latitude of point

        Returns
        -------
        other_df : pandas.DataFrame or None
            Other dataframe.
        """
        other = self.datasets[other_name]
        args = list(args)
        args.extend(other['args'])
        try:
            other_df = other['class'].read_ts(*args, **other['kwargs'])
        except IOError:
            warnings.warn(
                "IOError while reading other dataset {:}".format(args))
            return None
        except RuntimeError as e:
            if e.args[0] == "No such file or directory":
                warnings.warn(
                    "IOError while reading other dataset {:}".format(args))
                return None
            else:
                raise e

        if len(other_df) == 0:
            warnings.warn("No data for other dataset".format(args))
            return None

        if self.data_prep is not None:
            other_df = self.data_prep.prep_other(other_df, other_name)

        if len(other_df) == 0:
            warnings.warn("No data for other dataset {:}".format(args))
            return None

        if isinstance(other_df, pd.DataFrame) == False:
            warnings.warn("Data is not a DataFrame {:}".format(args))
            return None

        if self.period is not None:
            other_df = other_df[self.period[0]:self.period[1]]

        if len(other_df) == 0:
            warnings.warn("No data for other dataset {:}".format(args))
            return None

        else:
            return other_df
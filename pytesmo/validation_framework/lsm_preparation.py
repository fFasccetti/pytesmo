import copy
import numpy as np
import pytesmo.metrics as metrics

import netCDF4
import pandas as pd
import pytesmo.timedate.julian as julian
import pytesmo.time_series.anomaly as anomaly

#import pytesmo.temporal_matching as temp_match
#import pytesmo.validation_framework.temporal_matchers_Ini as temporal_matchers

class BasicMetricsTC(object):
    """
    This class just computes basic metrics for all the possible combinations and the results of the triple collocation.
    The first dataset is used as reference
    It also stores information about gpi, lat, lon
    """

    def __init__(self):

        metrics = {'snr_x': np.float32([np.nan]),
                   'snr_y': np.float32([np.nan]),
                   'snr_z': np.float32([np.nan]),
                   'err_x': np.float32([np.nan]),
                   'err_y': np.float32([np.nan]),
                   'err_z': np.float32([np.nan]),
                   'beta_x': np.float32([np.nan]),
                   'beta_y': np.float32([np.nan]),
                   'beta_z': np.float32([np.nan]),
                   'R_xy': np.float32([np.nan]),
                   'p_R_xy': np.float32([np.nan]),
                   'rho_xy': np.float32([np.nan]),
                   'p_rho_xy': np.float32([np.nan]),
                   'BIAS_xy': np.float32([np.nan]),
                   'RMSD_xy': np.float32([np.nan]),
                   'ubRMSD_xy': np.float32([np.nan]),
                   'R_xz': np.float32([np.nan]),
                   'p_R_xz': np.float32([np.nan]),
                   'rho_xz': np.float32([np.nan]),
                   'p_rho_xz': np.float32([np.nan]),
                   'BIAS_xz': np.float32([np.nan]),
                   'RMSD_xz': np.float32([np.nan]),
                   'ubRMSD_xz': np.float32([np.nan]),
                   'R_yz': np.float32([np.nan]),
                   'p_R_yz': np.float32([np.nan]),
                   'rho_yz': np.float32([np.nan]),
                   'p_rho_yz': np.float32([np.nan]),
                   'BIAS_yz': np.float32([np.nan]),
                   'RMSD_yz': np.float32([np.nan]),
                   'ubRMSD_yz': np.float32([np.nan]),
                   'n_obs': np.int32([0])}


        self.result_template = {'gpi': np.int32([-1]),
                                'lon': np.float32([np.nan]),
                                'lat': np.float32([np.nan])}

        self.seasons = ['ALL', 'DJF', 'MAM', 'JJA','SON']

        for season in self.seasons:
            for metric in metrics.keys():
                key = '{:}_{:}'.format(season,metric)
                self.result_template[key] = metrics[metric].copy()

        self.month_to_season = np.array(['','DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'])


    def calc_metrics(self,data,gpi_info):

        """
        Calculates the desired statistics

        Parameters:
        -----------
        data : pandas.DataFrame
               three columns named ref other1 other2
        gpi_info : tuple
                Grid Point Index (i.e., gpi lon lat)
        """
        dataset = copy.deepcopy(self.result_template)
        dataset['gpi'][0] = gpi_info[0]
        dataset['lon'][0] = gpi_info[1]
        dataset['lat'][0] = gpi_info[2]



        for season in self.seasons:

            if season != 'ALL':
                subset =  self.month_to_season[data.index.month] == season
            else:
                subset = np.ones(len(data),dtype=bool)

            if subset.sum() < 2:
                continue

            x,y,z = data['ref'].values[subset], data['other1'].values[subset], data['other2'].values[subset]

            R_xy, p_R_xy = metrics.pearsonr(x, y)
            rho_xy, p_rho_xy = metrics.spearmanr(x, y)
            RMSD_xy = metrics.rmsd(x, y)
            BIAS_xy = metrics.bias(x, y)
            ubRMSD_xy = metrics.ubrmsd(x, y)

            R_xz, p_R_xz = metrics.pearsonr(x, z)
            rho_xz, p_rho_xz = metrics.spearmanr(x, z)
            RMSD_xz = metrics.rmsd(x, z)
            BIAS_xz = metrics.bias(x, z)
            ubRMSD_xz = metrics.ubrmsd(x, z)

            R_yz, p_R_yz = metrics.pearsonr(y, z)
            rho_yz, p_rho_yz = metrics.spearmanr(y, z)
            RMSD_yz = metrics.rmsd(y, z)
            BIAS_yz = metrics.bias(y, z)
            ubRMSD_yz = metrics.ubrmsd(y, z)

            snr,err, beta = metrics.tcol_snr(x,y,z)

            dataset['{:}_n_obs'.format(season)][0] = subset.sum()

            dataset['{:}_R_xy'.format(season)][0] = R_xy
            dataset['{:}_p_R_xy'.format(season)][0] = p_R_xy
            dataset['{:}_rho_xy'.format(season)][0] = rho_xy
            dataset['{:}_p_rho_xy'.format(season)][0] = p_rho_xy
            dataset['{:}_RMSD_xy'.format(season)][0] = RMSD_xy
            dataset['{:}_ubRMSD_xy'.format(season)][0] = ubRMSD_xy
            dataset['{:}_BIAS_xy'.format(season)][0] = BIAS_xy

            dataset['{:}_R_xz'.format(season)][0] = R_xz
            dataset['{:}_p_R_xz'.format(season)][0] = p_R_xz
            dataset['{:}_rho_xz'.format(season)][0] = rho_xz
            dataset['{:}_p_rho_xz'.format(season)][0] = p_rho_xz
            dataset['{:}_RMSD_xz'.format(season)][0] = RMSD_xz
            dataset['{:}_ubRMSD_xz'.format(season)][0] = ubRMSD_xz
            dataset['{:}_BIAS_xz'.format(season)][0] = BIAS_xz

            dataset['{:}_R_yz'.format(season)][0] = R_yz
            dataset['{:}_p_R_yz'.format(season)][0] = p_R_yz
            dataset['{:}_rho_yz'.format(season)][0] = rho_yz
            dataset['{:}_p_rho_yz'.format(season)][0] = p_rho_yz
            dataset['{:}_RMSD_yz'.format(season)][0] = RMSD_yz
            dataset['{:}_ubRMSD_yz'.format(season)][0] = ubRMSD_yz
            dataset['{:}_BIAS_yz'.format(season)][0] = BIAS_yz

            dataset['{:}_snr_x'.format(season)][0] = snr[0]
            dataset['{:}_snr_y'.format(season)][0] = snr[1]
            dataset['{:}_snr_z'.format(season)][0] = snr[2]
            dataset['{:}_err_x'.format(season)][0] = err[0]
            dataset['{:}_err_y'.format(season)][0] = err[1]
            dataset['{:}_err_z'.format(season)][0] = err[2]
            dataset['{:}_beta_x'.format(season)][0] = beta[0]
            dataset['{:}_beta_y'.format(season)][0] = beta[1]
            dataset['{:}_beta_z'.format(season)][0] = beta[2]

        return dataset

class DataPreparationTC(object):

        # Here it is possible to convert ERA in m3/m3 or to convert ERA and SMOS in %

    def prep_reference(self,reference,reference_name='SMOS'):
        """
        Static method used to prepare the reference dataset SMOS.
        Parameters
        ----------
        reference : pandas.DataFrame
            Containing at least the fields: sm.

        Returns
        -------
        reference : pandas.DataFrame
            Masked reference.
        """

        if reference_name=='SMOS':
            reference = reference[reference['Soil_Moisture'] >= 0]
            reference = reference[reference['Soil_Moisture_DQX']<=0.045]
            reference = reference[reference['Soil_Moisture']<=0.7]

             # reference['Soil']

            # convertion of smos GP index to Pandas
            if reference.size>0:
                day = reference['jd']- julian.julday(01,01,1900)
                day = netCDF4.num2date(day,'days since 1900-01-01 00:00:00')

                sm = reference['Soil_Moisture']*100
                dict_sm_smos = {'Soil_Moisture': sm}
                df_smos = pd.DataFrame(dict_sm_smos, index=day)
                reference = df_smos

                # Anomalies
                reference['Soil_Moisture'] = anomaly.calc_anomaly(reference['Soil_Moisture'],window_size=35)

        return reference

    def prep_other(self,other, other_name, mask_ssf=[0,1]):

        """

        Static method used to prepare the other datasets (ERA and EraInterimLand).

        Parameters
        ----------
        other : pandas.DataFrame

        other_name : string
            ERS or EraIinterimLand.

        mask_ssf : list, optional ONLY for ERS
            If set, all the observations with ssf != mask_ssf are removed from
            the result. Default: [0, 1].

        Returns
        -------
        other : pandas.DataFrame
            Masked other.
        """

        if other_name == 'ERA':
            # mask for negative value of soil moisture
            other = other[other['sm_era'] >= 0]
            other = other[other['sm_era'] <= 0.7]
            other['sm_era'] = other['sm_era']*100
            other = pd.DataFrame(other)

            # Anomalies
            other['sm_era'] = anomaly.calc_anomaly(other['sm_era'],window_size=35)

        if other_name == 'ERS' or other_name == 'ASCAT':
            if mask_ssf is not None:
                # other = other[(other['ssf']== mask_ssf[0]) | (other['ssf'] == mask_ssf[1])]
                other = other[other['sm']>=0]
                # other['corr_flag'][np.isnan(other['corr_flag'])]=0
                # other['proc_flag'][np.isnan(other['proc_flag'])]=0
                corr_flag = np.int64(other['corr_flag'])
                corr_flag[corr_flag<0] = 0
                proc_flag = np.int64(other['proc_flag'])
                proc_flag[proc_flag<0] = 0
                bit_mask = ((get_bit(corr_flag, 3)) |
                            (get_bit(corr_flag, 4)) |
                            (get_bit(corr_flag, 6)) |
                            (get_bit(proc_flag, 1)) |
                            (get_bit(corr_flag, 2)))

                other = other[(other['ssf'] == 0) | (other['ssf'] == 1) &
                              (bit_mask == 0)]

                other = other[['sm','ssf']]
                # Anomalies
                other['sm'] = anomaly.calc_anomaly(other['sm'],window_size=35)

        return other


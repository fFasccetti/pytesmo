import os
import copy
import numpy as np
import pytesmo.metrics as metrics

import netCDF4
import pandas as pd
import pytesmo.timedate.julian as julian
import pytesmo.time_series.anomaly as anomaly

import rsdata.root_path as root


# import pytesmo.temporal_matching as temp_match
# import pytesmo.validation_framework.temporal_matchers as temporal_matchers

# def calc_por(gpi_info):
#     import os
#     from netCDF4 import Dataset
#     import rsdata.root_path as root
#     file_por = os.path.join(root.r,'Datapool_processed','WARP','ancillary','static_layer','static_layer_porosity.nc')
#     nc_por = Dataset(file_por,'r')
#     gpi_por = nc_por['location_id'][:]
#     porGLDAS = nc_por['por_gldas'][:]
#     lut_por_name = os.path.join('/home/','ffascett','Desktop','lut_POR.nc')
#     lut_por_nc = Dataset(lut_por_name,'r')
#     lut_por = lut_por_nc['location_id'][:]
#     por = porGLDAS._data[gpi_por==lut_por[gpi_info]][0]
#     return por

def get_bit(a, bit_pos):
    """
    Returns 1 or 0 if bit is set or not.

    Parameters
    ----------
    a : int or numpy.ndarray
      Input array.
    bit_pos : int
      Bit position. First bit position is right.

    Returns
    -------
    b : numpy.ndarray
      1 if bit is set and 0 if not.
    """
    return np.clip(np.bitwise_and(a, 2 ** (bit_pos-1)), 0, 1)

class BasicMetricsQC(object):
    """
    This class just computes basic metrics for all the possible combinations and the results of the triple collocation.
    The first dataset is used as reference
    It also stores information about gpi, lat, lon
    """

    def __init__(self):

        metrics = {'err_x': np.float32([np.nan]),
                   'err_y': np.float32([np.nan]),
                   'err_z': np.float32([np.nan]),
                   'err_w': np.float32([np.nan]),
                   'beta_x': np.float32([np.nan]),
                   'beta_y': np.float32([np.nan]),
                   'beta_z': np.float32([np.nan]),
                   'beta_w': np.float32([np.nan]),
                   'sigma': np.float32([np.nan]),
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
                   'R_xw': np.float32([np.nan]),
                   'p_R_xw': np.float32([np.nan]),
                   'rho_xw': np.float32([np.nan]),
                   'p_rho_xw': np.float32([np.nan]),
                   'BIAS_xw': np.float32([np.nan]),
                   'RMSD_xw': np.float32([np.nan]),
                   'ubRMSD_xw': np.float32([np.nan]),
                   'R_yw': np.float32([np.nan]),
                   'p_R_yw': np.float32([np.nan]),
                   'rho_yw': np.float32([np.nan]),
                   'p_rho_yw': np.float32([np.nan]),
                   'BIAS_yw': np.float32([np.nan]),
                   'RMSD_yw': np.float32([np.nan]),
                   'ubRMSD_yw': np.float32([np.nan]),
                   'R_zw': np.float32([np.nan]),
                   'p_R_zw': np.float32([np.nan]),
                   'rho_zw': np.float32([np.nan]),
                   'p_rho_zw': np.float32([np.nan]),
                   'BIAS_zw': np.float32([np.nan]),
                   'RMSD_zw': np.float32([np.nan]),
                   'ubRMSD_zw': np.float32([np.nan]),
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

        # load porosity infromation
        file_por = os.path.join(root.r,'Datapool_processed','WARP',
                                'ancillary','static_layer',
                                'static_layer_porosity.nc')
        with netCDF4.Dataset(file_por,'r') as nc_por:
            self.gpi_por = nc_por['location_id'][:]
            self.porGLDAS = nc_por['por_gldas'][:]
        lut_por_name = '/data-write/RADAR/Validation_FFascetti/lut_POR.nc'
        with netCDF4.Dataset(lut_por_name, 'r') as lut_por_nc:
            self.lut_por = lut_por_nc['location_id'][:]
 
        
    def calc_por(self, gpi_info):
        por = self.porGLDAS._data[self.gpi_por==self.lut_por[gpi_info]][0]
        return por


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

        por = self.calc_por(gpi_info[0])

        for season in self.seasons:

            if season != 'ALL':
                subset =  self.month_to_season[data.index.month] == season
            else:
                subset = np.ones(len(data),dtype=bool)

            if subset.sum() < 2: # 10
                continue

            x,y,z,w = data['ref'].values[subset], data['other1'].values[subset], data['other2'].values[subset], data['other3'].values[subset]

            if por==-1:
                continue
            else:
                w = w*por

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

            R_xw, p_R_xw = metrics.pearsonr(x, w)
            rho_xw, p_rho_xw = metrics.spearmanr(x, w)
            RMSD_xw = metrics.rmsd(x, w)
            BIAS_xw = metrics.bias(x, w)
            ubRMSD_xw = metrics.ubrmsd(x, w)

            R_yw, p_R_yw = metrics.pearsonr(y, w)
            rho_yw, p_rho_yw = metrics.spearmanr(y, w)
            RMSD_yw = metrics.rmsd(y, w)
            BIAS_yw = metrics.bias(y, w)
            ubRMSD_yw = metrics.ubrmsd(y, w)

            R_zw, p_R_zw = metrics.pearsonr(z, w)
            rho_zw, p_rho_zw = metrics.spearmanr(z, w)
            RMSD_zw = metrics.rmsd(z, w)
            BIAS_zw = metrics.bias(z, w)
            ubRMSD_zw = metrics.ubrmsd(z, w)

            sigma, err, beta = qcol(x,y,z,w)

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

            dataset['{:}_R_xw'.format(season)][0] = R_xw
            dataset['{:}_p_R_xw'.format(season)][0] = p_R_xw
            dataset['{:}_rho_xw'.format(season)][0] = rho_xw
            dataset['{:}_p_rho_xw'.format(season)][0] = p_rho_xw
            dataset['{:}_RMSD_xw'.format(season)][0] = RMSD_xw
            dataset['{:}_ubRMSD_xw'.format(season)][0] = ubRMSD_xw
            dataset['{:}_BIAS_xw'.format(season)][0] = BIAS_xw

            dataset['{:}_R_yw'.format(season)][0] = R_yw
            dataset['{:}_p_R_yw'.format(season)][0] = p_R_yw
            dataset['{:}_rho_yw'.format(season)][0] = rho_yw
            dataset['{:}_p_rho_yw'.format(season)][0] = p_rho_yw
            dataset['{:}_RMSD_yw'.format(season)][0] = RMSD_yw
            dataset['{:}_ubRMSD_yw'.format(season)][0] = ubRMSD_yw
            dataset['{:}_BIAS_yw'.format(season)][0] = BIAS_yw

            dataset['{:}_R_zw'.format(season)][0] = R_zw
            dataset['{:}_p_R_zw'.format(season)][0] = p_R_zw
            dataset['{:}_rho_zw'.format(season)][0] = rho_zw
            dataset['{:}_p_rho_zw'.format(season)][0] = p_rho_zw
            dataset['{:}_RMSD_zw'.format(season)][0] = RMSD_zw
            dataset['{:}_ubRMSD_zw'.format(season)][0] = ubRMSD_zw
            dataset['{:}_BIAS_zw'.format(season)][0] = BIAS_zw

            dataset['{:}_err_x'.format(season)][0] = err[0]
            dataset['{:}_err_y'.format(season)][0] = err[1]
            dataset['{:}_err_z'.format(season)][0] = err[2]
            dataset['{:}_err_w'.format(season)][0] = err[3]
            dataset['{:}_beta_x'.format(season)][0] = beta[0]
            dataset['{:}_beta_y'.format(season)][0] = beta[1]
            dataset['{:}_beta_z'.format(season)][0] = beta[2]
            dataset['{:}_beta_w'.format(season)][0] = beta[3]
            dataset['{:}_sigma'.format(season)][0] = sigma #[0]

        return dataset

def qcol(x,y,z,w):

    # Reference
    # Pierdicca, N., Fascetti, F., Pulvirenti, L., Crapolicchio R., Munoz-Sabater, J., "Quadruple Collocation Analysis for Soil Moisture Product Assessment",
    # IEEE Geoscience and Remote Sensing Letters, vol. 12, no. 8, August 2015

    """
    Parameters
    ----------
    x: 1D numpy.ndarray
      System used as reference for the Quadruple Collocation
    y: 1D numpy.ndarray
    z: 1D numpy.ndarray
    w: 1D numpy.ndarray

    Outputs
    -------
    sigma : numpy.ndarray
         Variance of the true variable
    err_std : numpy.ndarray
           error variance of x, y, z and systems, in the scale of the reference
    beta : numpy.ndarray
         gain of the system y,z and w respect to the reference
    """
    cov_matrix = np.cov(np.stack((x,y,z,w)))
    a = cov_matrix[0][3]/cov_matrix[1][3]
    b = cov_matrix[0][2]/cov_matrix[1][2]
    c = cov_matrix[0][1]/cov_matrix[1][2]
    d = cov_matrix[0][3]/cov_matrix[2][3]
    e = cov_matrix[0][1]/cov_matrix[1][3]
    f = cov_matrix[0][2]/cov_matrix[2][3]

    beta_y = (a+b)/(a**2+b**2)
    beta_z = (c+d)/(c**2+d**2)
    beta_w = (e+f)/(e**2+f**2)

    ratio1 = beta_y*cov_matrix[0][1]+beta_z*cov_matrix[0][2]+beta_w*cov_matrix[0][3]+beta_y*beta_z*cov_matrix[1][2]\
             +beta_y*beta_w*cov_matrix[1][3]+beta_z*beta_w*cov_matrix[2][3]
    ratio2 = beta_y**2+beta_z**2+beta_w**2+beta_y**2*beta_z**2+beta_y**2*beta_w**2+beta_z**2*beta_w**2

    sigma = ratio1/ratio2
    ex = cov_matrix[0][0]-sigma
    ey = cov_matrix[1][1]*beta_y**(-2)-sigma
    ez = cov_matrix[2][2]*beta_z**(-2)-sigma
    ew = cov_matrix[3][3]*beta_w**(-2)-sigma

    err_std = [ex, ey, ez, ew]
    beta = [1, beta_y, beta_z, beta_w]
    return sigma, err_std, beta


class DataPreparationQC(object):

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
                bit_mask = ((get_bit(other['corr_flag'],3))|(get_bit(other['corr_flag'],4))|(get_bit(other['corr_flag'],6))|(get_bit(other['proc_flag'],1)|(get_bit(other['corr_flag'],2))))
                other = other[((other['ssf'] == 0)|(other['ssf'] == 1))&(bit_mask == 0)]
                other = other[other['sm']>=0]

                other = other[['sm','ssf']]
                # Anomalies
                other['sm'] = anomaly.calc_anomaly(other['sm'],window_size=35)
        if other_name == 'AMSRE':

            jday = other['jd'] - julian.julday(01,01,1900)
            jday = netCDF4.num2date(jday, 'days since 1900-01-01 00:00:00')
            other = pd.DataFrame({'smc': other['smc'],'smx':other['smx'], 'dir':other['dir']}, index=jday)
            other = other[other['dir']==68]
            other = other[other['smc']>=0]
            other = other[other['smc']<=70]
            other = other[['smc']]
            # Anomalies
            other['smc'] = anomaly.calc_anomaly(other['smc'],window_size=35)
        return other
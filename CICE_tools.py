# Codes for manipulating CICE output
import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
from mpl_toolkits.basemap import Basemap
from sys import path
from netCDF4 import Dataset
# path.insert(0, '/Users/H/WAVES/geo_data_group/')
import data_year as dy
import grid_set as gs
import copy

# 

def CICE2dy(hist_dir,var_name,time_start,time_end,write_dir='./',
            t_p='months',n_p=1,time_type='',vectors=False,return_dy = False,write_dy = True):
    """
    code to fly through CICE histories and make a dy from many time slices from CICE files
    options should cover most of the history file opitons for CICE
    """
#     # now I want a CICE2dy script
#     # define directory to read from
#     hist_dir = '/Users/H/RHEOLOGY/CICE/Histories/Convergence_tests_1deg/history_EAP_Forig_1800_120/'

#     # variable name
#     var_name = 'uvel'

#     vectors = True

#     var_name = ('uvel','vvel')

#     # file type we're reading - inst or av
#     time_type = '_inst'
#     # time_type = ''

#     # time period
#     t_p = 'months'
#     t_p = 'days'
#     t_p = 'seconds'

#     n_p = 3600


#     time_start = dt.datetime(1980,1,1)+relativedelta(seconds=1800)
#     time_end   = dt.datetime(1980,1,1)+relativedelta(seconds=5400)

    # define write file
    if vectors:
        dy_file = write_dir+var_name[0]+var_name[1]+'_'+t_p+time_start.strftime('_%Y-%m-%d')+time_end.strftime('_%Y-%m-%d')+'_CICE_dy.npz'
    else:
        dy_file = write_dir+var_name+'_'+t_p+time_start.strftime('%_Y-%m-%d')+'_'+time_end.strftime('%_Y-%m-%d')+'_CICE_dy.npz'

    # set time points - depends on t_p
    if t_p =='months':
        pp =(time_end.year - time_start.year) * 12 + time_end.month-time_start.month
        pp = int(pp/n_p) + 1
        dy_periods = int(12/n_p) 

    if t_p =='days':
        pp =(time_end - time_start).days
        pp = int(pp/n_p) + 1
        dy_periods = int(366/n_p) 

    if t_p =='seconds':
        pp =(time_end - time_start).seconds
        pp = int(pp/n_p) + 1
        dy_periods = int(366*24*3600/n_p) 

    # open first file and get shape
    # inst - needs up to seconds
    # iceh_inst.1980-01-01-01800.nc
    midnight = time_start.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (time_start - midnight).seconds
    ### [filename_ic]=[directory,'iceh_ic.',datestr(sd,'yyyy-mm'),'-01-00000.nc'];
    f1 = 'iceh_ic'+time_start.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'

    time_use = time_start
    if t_p =='months':
        f = 'iceh'+time_use.strftime('.%Y-%m')+'.nc'
    if t_p =='days':
        f = 'iceh'+time_use.strftime('.%Y-%m-%d')+'.nc'
    if t_p =='seconds':
        f = 'iceh'+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
    if time_type == '_inst':
        f = 'iceh'+time_type+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'

    # f1 = 'iceh'+time_type+time_start.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
    ncf = Dataset(hist_dir+f)
    ni = ncf.dimensions['ni'].size
    nj = ncf.dimensions['nj'].size
    ncf.close()
    if vectors:
        datax = np.empty([pp,ni,nj])
        datay = np.empty([pp,ni,nj])
    else:
        data = np.empty([pp,ni,nj])
    mask = np.empty([pp,ni,nj],dtype=bool)
    dates = []

    # now loop and fill array and dates list
    for t in range(pp):
        if t_p =='months':
            time_use = time_start + relativedelta(months=t*n_p)
            f = 'iceh'+time_use.strftime('.%Y-%m')+'.nc'
        if t_p =='days':
            time_use = time_start + relativedelta(days=t*n_p)
            f = 'iceh'+time_use.strftime('.%Y-%m-%d')+'.nc'
        if t_p =='seconds':
            time_use = time_start + relativedelta(seconds=t*n_p)
            midnight = time_use.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds = (time_use - midnight).seconds
            f = 'iceh'+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
        if time_type == '_inst':
            f = 'iceh'+time_type+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
        print(f)
        ncf = Dataset(hist_dir+f)
        if vectors:
            datax[t] = ncf[var_name[0]][:].data[0].T
            datay[t] = ncf[var_name[1]][:].data[0].T
            mask[t] = ncf[var_name[0]][:].mask[0].T == False
        else:
            data[t] = ncf[var_name][:].data[0].T
            mask[t] = ncf[var_name][:].mask[0].T == False
        ncf.close()
        dates.append(time_use)

    if vectors:
        out_dy = dy.vec_data_year(datax,datay,dates,periods=dy_periods) 
    else:
        out_dy = dy.data_year(data,dates,periods=dy_periods) 
    out_dy.mask = mask
    if 'mask_nan' and vectors:
        out_dy.x[out_dy.mask == False] = np.nan
        out_dy.y[out_dy.mask == False] = np.nan
    elif 'mask_nan':
        out_dy.data[out_dy.mask == False] = np.nan
    if write_dy:
        out_dy.save(dy_file)
        
    if return_dy:
        return out_dy




def CICE2dy_regrid(hist_dir,var_name,time_start,time_end,Gs2Gs,write_dir='./',
            t_p='months',n_p=1,time_type='',
            vectors=False,return_dy = False,write_dy = True,m_lim=0.0,mask_nan=False):
    """
    code to fly through CICE histories and make a dy from many time slices from CICE files
    options should cover most of the history file opitons for CICE
    
    The functions adds on addtional regridding scripts - developed for square grid regional evaluations
    You need to previously define a Gs2Gs object using grid_set
    0<m_lim<1 is the parameter for regridding the mask
        = 0 is all cells that are touched by the original
        = 1 is only those covered by the original
    """
    # define write file
    if vectors:
        dy_file = write_dir+var_name[0]+var_name[1]+'_'+t_p+time_start.strftime('_%Y-%m-%d')+time_end.strftime('_%Y-%m-%d')+'_CICE_dyregrid.npz'
    else:
        dy_file = write_dir+var_name+'_'+t_p+time_start.strftime('_%Y-%m-%d')+time_end.strftime('_%Y-%m-%d')+'_CICE_dyregrid.npz'

    # set time points - depends on t_p
    if t_p =='months':
        pp =(time_end.year - time_start.year) * 12 + time_end.month-time_start.month
        pp = int(pp/n_p) + 1
        dy_periods = int(12/n_p) 

    if t_p =='days':
        pp =(time_end - time_start).days
        pp = int(pp/n_p) + 1
        dy_periods = int(366/n_p) 

    if t_p =='seconds':
        pp =(time_end - time_start).seconds
        pp = int(pp/n_p) + 1
        dy_periods = int(366*24*3600/n_p) 

    # open first file and get shape
    # inst - needs up to seconds
    # iceh_inst.1980-01-01-01800.nc
    midnight = time_start.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds = (time_start - midnight).seconds
    ### [filename_ic]=[directory,'iceh_ic.',datestr(sd,'yyyy-mm'),'-01-00000.nc'];
    f1 = 'iceh_ic'+time_start.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'

    time_use = time_start
    if t_p =='months':
        f = 'iceh'+time_use.strftime('.%Y-%m')+'.nc'
    if t_p =='days':
        f = 'iceh'+time_use.strftime('.%Y-%m-%d')+'.nc'
    if t_p =='seconds':
        f = 'iceh'+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
    if time_type == '_inst':
        f = 'iceh'+time_type+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'

    # f1 = 'iceh'+time_type+time_start.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
    # this time the data array is set by the new grid
    ni,nj = np.shape(Gs2Gs.mesh_new)[1:3]
    
    if vectors:
        datax = np.empty([pp,ni,nj])
        datay = np.empty([pp,ni,nj])
        datax[:] = np.nan
        datay[:] = np.nan
    else:
        data = np.empty([pp,ni,nj])
        data[:] = np.nan
    mask = np.empty([pp,ni,nj],dtype=bool)
    dates = []

    # now loop and fill array and dates list
    for t in range(pp):
        if t_p =='months':
            time_use = time_start + relativedelta(months=t*n_p)
            f = 'iceh'+time_use.strftime('.%Y-%m')+'.nc'
        if t_p =='days':
            time_use = time_start + relativedelta(days=t*n_p)
            f = 'iceh'+time_use.strftime('.%Y-%m-%d')+'.nc'
        if t_p =='seconds':
            time_use = time_start + relativedelta(seconds=t*n_p)
            midnight = time_use.replace(hour=0, minute=0, second=0, microsecond=0)
            seconds = (time_use - midnight).seconds
            f = 'iceh'+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
        if time_type == '_inst':
            f = 'iceh'+time_type+time_use.strftime('.%Y-%m-%d-')+"{:05}".format(seconds)+'.nc'
        print(f)
        ncf = Dataset(hist_dir+f)
        if vectors:
            inx  = ncf[var_name[0]][0,:,:]
            iny  = ncf[var_name[1]][0,:,:]
            inx[inx.mask] = np.nan
            iny[iny.mask] = np.nan
            datax[t],datay[t] = Gs2Gs.rg_vecs(
                 inx.T,
                 iny.T)
            mask[t] =  Gs2Gs.rg_array(
                ncf[var_name[0]][:].mask[0].T == False )>m_lim
        else:
            ind  = ncf[var_name][0,:,:]
            ind[ind.mask] = np.nan
            data[t] =  Gs2Gs.rg_array(
                ind.T)
            mask[t] =  Gs2Gs.rg_array(
                ncf[var_name][:].mask[0].T == False )>m_lim
        ncf.close()
        dates.append(time_use)

    if vectors:
        out_dy = dy.vec_data_year(datax,datay,dates,periods=dy_periods) 
    else:
        out_dy = dy.data_year(data,dates,periods=dy_periods) 
    out_dy.mask = mask
    if mask_nan and vectors:
        out_dy.x[out_dy.mask == False] = np.nan
        out_dy.y[out_dy.mask == False] = np.nan
    elif mask_nan:
        out_dy.data[out_dy.mask == False] = np.nan
        
    if write_dy:
        out_dy.save(dy_file)
        
    if return_dy:
        return out_dy



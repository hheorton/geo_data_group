# here are the classes for holding data
# the first is a class that is a single gridded array
# it's class so that we can access the time variables happily
# first data class is a data_month
# in the dimension [time,x,y] format
# why a class not just any old array?
# so we can have the methods that take averages according to the time scale we want
# yearlies... monthlies... runnning means... all nicely indexed

import numpy as np
import pandas as pd
import datetime as dt
import shutil
import os
import copy
from invoke import run
from numba import jit
from scipy import stats
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap


class data_year:
    # this will init with an input array
    # take the datetime time dims
    # construct a yrpd masked array to allow indexing
    def __init__(self,data,dates,periods = 12):
        """
        data map all tidy like
        initialise with how many data points per year
        default is periods = 12 (monthly)
        the data is the data - just links the object to methods 
        # the date are a list of datetimes, seperated by months
        it's up to the user to supply a list of datetimes, spaced by the correct periods
        ----------
        the datetimes and periods need to comply, if you want 3 monthly seasons
        make sure the datetime lists work for this
        likewise if you have daily data, give the correct periods,
        or data every 10 days, etc etc
        """
        # check if the data timeis the same shape as the date
        # get range of years from dates list
        # 
        [n_t,m,n] = np.shape(data)
        self.n_t = n_t
        self.m = m
        self.n = n
        self.nyrs = dates[-1].year - dates[0].year + 1 
        self.yrpd = np.ma.empty([self.nyrs,periods],dtype = int)
        self.yrpd[:,:] = -1
        self.periods = periods
        self.data = data
        self.mask = False
        self.dates = dates
        self.saved = False
        
        if periods == 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                mt = self.dates[tt].month - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        elif periods < 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                mt = int(self.dates[tt].month*periods/12) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        elif periods > 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                day_of_year = (self.dates[tt] - dt.datetime(self.dates[tt].year,1,1)).days + 1
                mt = int(day_of_year*periods/365) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        self.yrpd.mask = self.yrpd < 0
        self.files = True
        

    def __getitem__(self,indx):
        t_p,m,n = indx
        if type(self.mask) == bool:
            # just return the data because it's not masked
            return self.data[t_p,m,n]
        else:
            temp = copy.copy(self.data[t_p,m,n])
            try: temp_mask = self.mask[t_p,m,n]
            except IndexError:
                return self.data[t_p,m,n]
            else: pass
        if type(temp) == np.ndarray:
            temp[temp_mask==False] = np.nan
            return temp
        elif temp_mask:
            return temp
        else:
            return np.nan
        
    def print_date(self,t,string='auto',year_only=False):
        """
        Quickly return a date lable from a given data_year time point
        return format can be overidden by setting string to datetime string format
        otherwise it is 'auto'
        year_only = true overides everything and just gives the year
        """
        # simply get a date string for a time point
        if year_only: # year_only overides
            str_option = '%Y'
        elif string=='auto':
            # auto generate the strftime option from no. of periods
            # if periods = 4 then year + JFM etc...
            # Add this option later use yrpd to find quarter 
                # manually set JFM etc
            # if periods < 12 then months only
            if self.periods <= 12:
                str_option = '%Y-%m'
            elif self.periods <= 366:
                str_option = '%Y-%m-%d'
            # longer then days too
            else:
                str_option = '%Y-%m-%d-T%H'
        else:
            str_option = string
        return self.dates[t].strftime(str_option)
 

    def build_mask(self):
        """
        fills the mask array with ones if it isn't there yet
        """
        if type(self.mask) == bool:
            self.mask = np.ones_like(self.data,dtype=bool)
        

    def build_static_mask(self,mask,points = False,overwrite=False):
        """
        Makes a satic 2d mask for all data
        or only the time points listed in points option
        mask is 1/True for good, nan/False bad
        overwrite = True, makes the mask identical to input mask
        overwrite = False, appends the current mask to match
        if you want to make a temporal mask for  a condition
        do it your self with logical
        ie.
        DY.build_mask()
        DY.mask[DY.data > limit] = 0
        will temporarily mask out all data  
        over the limit
        """
        if (type(mask[0,0])!=bool) and (type(mask[0,0])!=np.bool_):
            print('mask needs to be binary, not',type(mask[0,0]))
            return
        if type(points) == bool:
            points = np.arange(0,self.n_t,dtype=int)
        if overwrite:
            temp_mask = np.ones_like(self.data,dtype=bool)
            for tt in points:
                temp_mask[tt][mask==False] = False
            self.mask = temp_mask
        else:
            for tt in points:
                self.mask[tt][mask==False] = False
        
    def append(self,date,data):
        # check if there's data here already
        if ~self.files:
            print("nothing here, so can't append")
            return False
        # check the new data is the correct size
        m_check,n_check = np.shape(data)
        if m_check==self.m&n_check==self.n:
        # append the data
            self.data = np.append(self.data,np.expand_dims(data,axis=0),axis = 0)
            self.dates.append(date)
        # find the final entry in the yrpd
            loc = np.where(self.yrpd == self.n_t)
        # add the next entry(ies)
            if loc[1][0] == self.periods - 1:
        # add new rows if needed - keep the yrmth consistent
                self.yrpd = np.ma.append(self.yrpd,
                            np.ma.masked_values(np.ones([1,self.periods]),1),axis=0)
                self.yrpd[-1,0] = self.n_t + 1
            else:
                self.yrpd[-1,loc[1][0]] = self.n_t + 1
            self.n_t += 1
            return True
        else: return True
        # adds another time slice to the data, and sorts out the yrpd array

    def clim_map(self,periods,mask = False,year_set = [],time_set = []):
        """
        periods is the list of period no.s to use in the map
        ie. periods = [0,1,2] with give the map of average over
        the first three months of a monthly data_year
        """
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            # build a year limit if it's wanted
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            # check if months is in the correct form
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            # check if months is in the correct form
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
#             temp_array = np.empty([self.m,self.n])
            idx = [self.yrpd[y0:yE+1,mn].compressed() for mn in periods]
            temp_array = np.nanmean([self.data[j,:,:]
                            for i in idx for j in i if j>=t0 and j<=tE],axis = 0)
            temp_mask = np.sum([mask[j,:,:]
                            for i in idx for j in i if j>=t0 and j<=tE],axis = 0)
            temp_array[temp_mask==False] = np.nan
            return temp_array


        
    def clim_mean(self,mask = False,year_set = [],time_set = [],method='mean'):
        """
        Mask needs to be 1 for true 0 for false
        """
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            # check if months is in the correct form
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            # check if months is in the correct form
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
            # only use the periods that have data
            temp_array = np.empty([self.periods])
            for mn in range(self.periods):
                idx = self.yrpd[y0:yE+1,mn].compressed()
                t_mn = np.sum((idx>=t0)&(idx<=tE))
                temp      = np.empty([t_mn,self.m,self.n])
                temp_mask = np.empty([t_mn,self.m,self.n],dtype=bool)
                temp[:,:,:]      = [self.data[i] for i in idx if i>=t0 and i<=tE]
                temp_mask[:,:,:] = [self.mask[i] for i in idx if i>=t0 and i<=tE]
                temp[temp_mask==False] = np.nan
                if method=='mean':
                    temp_array[mn] = np.nanmean(temp)
                elif method=='median':
                    temp_array[mn] = np.nanmedian(temp)
            return temp_array

    def ravel(self,mask = False,periods=[],year_set = [],time_set = []):
        """
        Mask needs to be 1 for true 0 for false
        """
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.m,self.n],dtype=bool)
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.m,self.n],dtype=bool)
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.m,self.n],dtype=bool)
#             else:
#                 for j in range(self.n_t):
#                     temp[j] = ma
#             # build a period limit if none given
            if np.size(periods)==0:
                periods = np.arange(0,self.periods)
#             print(periods)
            # build a time limit if it's wanted
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            # check if months is in the correct form
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            # check if months is in the correct form
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
            temp_mask = np.zeros([self.n_t,self.m,self.n],dtype=bool)
            idx = [self.yrpd[y0:yE+1,mn].compressed() for mn in periods]
#             print(idx)
            for i in idx:
                for j in i:
                    if j>=t0 and j<=tE:
                        temp_mask[j]=mask[j]
            return self.data[temp_mask]



    def save(self,filename):
        """saves all the DY in a npz file"""
        if not self.saved:
            temp_time = [self.dates[t].toordinal() for t in range(self.n_t)]
            np.savez(filename,
                DY_data = self.data,
                DY_mask = self.mask,
                DY_yrpd = self.yrpd,
                DY_n_t  = self.n_t,
                DY_m = self.m,
                DY_n = self.n,
                DY_nyrs = self.nyrs ,
                DY_periods = self.periods ,
                DY_dates = temp_time)
            self.saved = True

# and now this will do the same but with vectors
class vec_data_year:
    # this will init with an input array
    # take the datetime time dims
    # construct a yrpd masked array to allow indexing
    def __init__(self,datax,datay,dates,periods = 12):
        """
        data map all tidy like
        initialise with how many data points per year
        default is periods = 12 (monthly)
        the data is the data - just links the object to methods 
        # the date are a list of datetimes, seperated by months
        it's up to the user to supply a list of datetimes, spaced by the correct periods
        ----------
        the datetimes and periods need to comply, if you want 3 monthly seasosn
        make sure the datetime lists work for this
        likewise if you have daily data, give the correct periods,
        or data every 10 days, etc etc
        """
        # check if the data timeis the same shape as the date
        # get range of years from dates list
        # 
        
        [n_t,m,n] = np.shape(datax)
        self.n_t = n_t
        self.m = m
        self.n = n
        self.saved = False
        self.nyrs = dates[-1].year - dates[0].year + 1 
        self.yrpd = np.ma.empty([self.nyrs,periods],dtype = int)
        self.mask = False
        self.yrpd[:,:] = -1
        self.periods = periods
        self.dates = dates
        self.x = datax
        self.y = datay
        if periods == 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                mt = self.dates[tt].month - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        elif periods < 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                mt = int(self.dates[tt].month*periods/12) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        elif periods > 12:
            for tt in range(self.n_t):
                yr = self.dates[tt].year - self.dates[0].year
                mt = int(self.dates[tt].day*periods/365) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        self.yrpd.mask = self.yrpd < 0
        self.files = True

    def __getitem__(self,indx):
        t_p,m,n = indx
        if type(self.mask) == bool:
            # just return the data because it's not masked
            return self.x[t_p,m,n], self.y[t_p,m,n]
        else:
            temp_x = self.x[t_p,m,n]
            temp_y = self.y[t_p,m,n]
            temp_mask = self.mask[t_p,m,n]
        if type(temp_x) == np.ndarray:
            temp_x[temp_mask==False] = np.nan
            temp_y[temp_mask==False] = np.nan
            return temp_x, temp_y
        elif temp_mask:
            return temp_x, temp_y
        else:
            return np.nan, np.nan
 
    def mag(self,indx):
        """
        crude getitem for magnitudes
        indx = [3,item,thing] to get the bits you want
        we cannot pass : 
        instead of ':' write 'slice(None)'
        instead of 'a:b' write 'slice(a,b,None)'
        """
        t_p,m,n = indx
        if type(self.mask) == bool:
            # just return the data because it's not masked
            return np.hypot(self.x[t_p,m,n], self.y[t_p,m,n])
        else:
            temp_x = self.x[t_p,m,n]
            temp_y = self.y[t_p,m,n]
            temp_mask = self.mask[t_p,m,n]
        if type(temp_x) == np.ndarray:
            temp_x[temp_mask==False] = np.nan
            temp_y[temp_mask==False] = np.nan
            return np.hypot(temp_x, temp_y)
        elif temp_mask:
            return np.hypot(temp_x, temp_y)
        else:
            return np.nan
    def print_date(self,t,string='auto',year_only=False):
        """
        Quickly return a date lable from a given data_year time point
        return format can be overidden by setting string to datetime string format
        otherwise it is 'auto'
        year_only = true overides everything and just gives the year
        """
        # simply get a date string for a time point
        if year_only: # year_only overides
            str_option = '%Y'
        elif string=='auto':
            # auto generate the strftime option from no. of periods
            # if periods = 4 then year + JFM etc...
            # Add this option later use yrpd to find quarter 
                # manually set JFM etc
            # if periods < 12 then months only
            if self.periods <= 12:
                str_option = '%Y-%m'
            elif self.periods <= 366:
                str_option = '%Y-%m-%d'
            # longer then days too
            else:
                str_option = '%Y-%m-%d-T%H'
        else:
            str_option = string
        return self.dates[t].strftime(str_option)
 


    def build_mask(self):
        """
        fills the mask array with ones if it isn't there yet
        """
        if type(self.mask) == bool:
            self.mask = np.ones_like(self.data,dtype=bool)
        


    def build_static_mask(self,mask,points = False,overwrite=False):
        """
        Makes a satic 2d mask for all data
        or only the time points listed in points option
        mask is 1/True for good, nan/False bad
        overwrite = True, makes the mask identical to input mask
        overwrite = False, apends the current mask to match
        if you want to make a temporal mask for  a condition
        do it your self with logical
        ie.
        DY.build_mask()
        DY.mask[DY.data > limit] = 0
        will temporarily mask out all data  
        over the limit
        """
        if (type(mask[0,0])!=bool) and (type(mask[0,0])!=np.bool_):
            print('mask needs to be binary, not',type(mask[0,0]))
            return
        if type(points) == bool:
            points = np.arange(0,self.n_t,dtype=int)
        if overwrite:
            temp_mask = np.ones_like(self.data,dtype=bool)
            for tt in points:
                temp_mask[tt][mask==False] = False
            self.mask = temp_mask
        else:
            for tt in points:
                self.mask[tt][mask==False] = False
        
    def append(self,date,datax,datay):
        # check if there's data here already
        if ~self.files:
            print("nothing here, so can't append")
            return False
        # check the new data is the correct size
        m_check,n_check = np.shape(datax)
        if m_check==self.m&n_check==self.n:
        # append the data
            self.x = np.append(self.x,np.expand_dims(datax,axis=0),axis = 0)
            self.y = np.append(self.y,np.expand_dims(datay,axis=0),axis = 0)
            self.dates.append(date)
        # find the final entry in the yrpd
            loc = np.where(self.yrpd == self.n_t)
        # add the next entry(ies)
            if loc[1][0] == self.periods - 1:
        # add new rows if needed - keep the yrmth consistent
                self.yrpd = np.ma.append(self.yrpd,
                            np.ma.masked_values(np.ones([1,self.periods]),1),axis=0)
                self.yrpd[-1,0] = self.n_t + 1
            else:
                self.yrpd[-1,loc[1][0]] = self.n_t + 1
            self.n_t += 1
            return True
        else: return True
        # adds another time slice to the data, and sorts out the yrpd array

    def clim_map(self,periods,mask = False,magnitude = False,year_set = [],time_set = []):
        """
        periods is the list of period no.s to use in the map
        ie. periods = [0,1,2] with give the map of average over
        the first three months of a monthly data_year
        setting magnitude = True, takes the average of the vector
        hypot, rather than the average of each component
        """ 
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            # now some year/time limits
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            # check if months is in the correct form
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            # check if months is in the correct form
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
            idx = [self.yrpd[y0:yE+1,mn].compressed() for mn in periods]
            temp_mask = np.sum([mask[j,:,:]
                            for i in idx for j in i if j>=t0 and j<=tE],axis = 0)
            if magnitude:
                temp_x = np.nanmean(
                [np.hypot(self.x[j],self.y[j]) for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                temp_x[temp_mask==False] = np.nan
                return temp_x
            else:
                temp_x = np.nanmean(
                    [self.x[j] for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                temp_y = np.nanmean(
                    [self.y[j] for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                temp_x[temp_mask==False] = np.nan
                temp_y[temp_mask==False] = np.nan
                return temp_x,temp_y


    def clim_mean(self,mask = False,magnitude = False,year_set = [],time_set = [],method='mean'):
        """
        Mask needs to be 1 for true 0 for false
        """
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            elif (np.shape(mask)[0] != self.m 
                   |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n],dtype=bool)
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
            temp_x = np.empty([self.periods])
            temp_y = np.empty([self.periods])
            if magnitude:
                for mn in range(self.periods):
                    idx = self.yrpd[y0:yE+1,mn].compressed()
                    t_mn = np.sum((idx>=t0)&(idx<=tE))
                    temp_x1   = np.empty([t_mn,self.m,self.n])
                    temp_y1   = np.empty([t_mn,self.m,self.n])
                    temp_mask = np.empty([t_mn,self.m,self.n],dtype=bool)
                    temp_x1[:,:,:]   = [self.x[i]    for i in idx if i>=t0 and i<=tE]
                    temp_y1[:,:,:]   = [self.y[i]    for i in idx if i>=t0 and i<=tE]
                    temp_mask[:,:,:] = [self.mask[i] for i in idx if i>=t0 and i<=tE]
                    temp = np.hypot(temp_x1,temp_y1)
                    temp[temp_mask==False] = np.nan
                    if method=='mean':
                        temp_x[mn] = np.nanmean(temp)
                    elif method=='median':
                        temp_x[mn] = np.nanmedian(temp)
                return temp_x
            else:
                for mn in range(self.periods):
                    idx = self.yrpd[y0:yE+1,mn].compressed()
                    t_mn = np.sum((idx>=t0)&(idx<=tE))
                    temp      = np.empty([t_mn,self.m,self.n])
                    temp_mask = np.empty([t_mn,self.m,self.n],dtype=bool)
                    temp[:,:,:]      = [self.x[i]    for i in idx if i>=t0 and i<=tE]
                    temp_mask[:,:,:] = [self.mask[i] for i in idx if i>=t0 and i<=tE]
                    temp[temp_mask==False] = np.nan
                    if method=='mean':
                        temp_x[mn] = np.nanmean(temp)
                    elif method=='median':
                        temp_x[mn] = np.nanmedian(temp)
                    temp[:,:,:]      = [self.y[i]    for i in idx if i>=t0 and i<=tE]
                    temp[temp_mask==False] = np.nan
                    if method=='mean':
                        temp_y[mn] = np.nanmean(temp)
                    elif method=='median':
                        temp_y[mn] = np.nanmedian(temp)
                return temp_x,temp_y

    def ravel(self,mask = False,magnitude = False,
              periods=[],year_set = [],time_set = []):
        """
        Mask needs to be 1 for true 0 for false
        """
        # check if there's data here already
        if self.files:
            if type(mask)==bool:
                if mask and type(self.mask)== bool:
                    print("Data year mask not set: ignoring it")
                    mask = np.ones([self.n_t,self.m,self.n])
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n])
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n])
            # build a period limit if none given
            if np.size(periods)==0:
                periods = np.arange(0,self.periods)
#             print(periods)
            # build a time limit if it's wanted
            if np.size(year_set)==0:
                y0 = 0
                yE = self.nyrs
            # check if months is in the correct form
            elif year_set[0]>self.nyrs:
                print("year_set inconistent with data, ignoring it")
                y0 = 0
                yE = self.nyrs
            else:
                y0 = year_set[0]
                yE = np.min([self.nyrs,year_set[1]])
            # build a time limit if it's wanted
            if np.size(time_set)==0:
                t0 = 0
                tE = self.n_t
            # check if months is in the correct form
            elif time_set[0]>self.n_t:
                print("time_set inconistent with data, ignoring it")
                t0 = 0
                tE = self.n_t
            else:
                t0 = time_set[0]
                tE = np.min([self.n_t,time_set[1]])
            temp_mask = np.zeros([self.n_t,self.m,self.n],dtype=bool)
            idx = [self.yrpd[y0:yE+1,mn].compressed() for mn in periods]
#             print(idx)
#             print(idx)
            for i in idx:
                for j in i:
                    if j>=t0 and j<=tE:
                        temp_mask[j]=mask[j]
            if magnitude:
                return np.hypot(self.x[temp_mask],self.y[temp_mask])
            else:
                return self.x[temp_mask],self.y[temp_mask]

    def save(self,filename):
        """saves all the DY in a npz file"""
        if not self.saved:
            temp_time = [self.dates[t].toordinal() for t in range(self.n_t)]
            np.savez(filename,
                DY_x = self.x,
                DY_y = self.y,
                DY_mask = self.mask,
                DY_yrpd = self.yrpd,
                DY_n_t  = self.n_t,
                DY_m = self.m,
                DY_n = self.n,
                DY_nyrs = self.nyrs ,
                DY_periods = self.periods ,
                DY_dates = temp_time)
            self.saved = True

    def build_mask(self):
        """
        fills the mask array with ones if it isn't there yet
        """
        if type(self.mask) == bool:
            self.mask = np.ones_like(self.x)
        
    def build_static_mask(self,mask,points = False):
        """
        Makes a satic 2d mask for all data
        or only the time points listed in points option
        mask is 1/True for good, nan/False bad
        if you want to make a temporal mask for  a condition
        do it your self with logical
        ie.
        DY.build_mask()
        DY.mask[DY.data > limit] = np.nan
        will temporally mask out all data  
        over the limit
        """
        self.build_mask()
        if type(points) == bool:
            points = np.arange(0,self.n_t,dtype=int)
        for tt in points:
            self.mask[tt] = mask




def load_data_year(filename):
    """
    give a save filename with a saved data year
    creates a data year from what was saved
    """
    try:
        npzfile =  np.load(filename)
        data = npzfile["DY_data"] 
    except KeyError:
        pass
    else:
        temp_time = npzfile["DY_dates"] 
        dates = [dt.datetime.fromordinal(t) for t in temp_time]
        DY_out = data_year(data,dates)
        DY_out.mask = npzfile["DY_mask"] 
        DY_out.periods = npzfile["DY_periods"] 
        DY_out.nyrs = npzfile["DY_nyrs"] 
        DY_out.yrpd = np.ma.empty([DY_out.nyrs,DY_out.periods],dtype = int)
        temp_yrpd = npzfile["DY_yrpd"] 
        npzfile.close()
        DY_out.yrpd[:,:] = temp_yrpd[:,:]
        DY_out.yrpd.mask = DY_out.yrpd < 0
#         DY.n_t = npzfile["DY_n_t"]  
#         self.m = npzfile["DY_m"] = m
#         self.n = npzfile["DY_n"] = n
        DY_out.saved = True
        return DY_out


def load_vec_data_year(filename):
    """
    give a save filename with a saved vector data year
    creates a vector data year from what was saved
    """
    try:
        npzfile =  np.load(filename)
        x = npzfile["DY_x"] 
        y = npzfile["DY_y"] 
    except KeyError:
        pass
    else:
        temp_time = npzfile["DY_dates"] 
        dates = [dt.datetime.fromordinal(t) for t in temp_time]
        DY_out = vec_data_year(x,y,dates)
        DY_out.mask = npzfile["DY_mask"] 
        DY_out.periods = npzfile["DY_periods"] 
        DY_out.nyrs = npzfile["DY_nyrs"] 
        DY_out.yrpd = np.ma.empty([DY_out.nyrs,DY_out.periods],dtype = int)
        temp_yrpd = npzfile["DY_yrpd"] 
        npzfile.close()
        DY_out.yrpd[:,:] = temp_yrpd[:,:]
        DY_out.yrpd.mask = DY_out.yrpd < 0
#         DY.n_t = npzfile["DY_n_t"]  
#         self.m = npzfile["DY_m"] = m
#         self.n = npzfile["DY_n"] = n
        DY_out.saved = True
        return DY_out
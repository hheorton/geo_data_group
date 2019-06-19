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
                mt = int(self.dates[tt].days*periods/365) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        self.yrpd.mask = self.yrpd < 0
        self.files = True
        
    def build_mask(self):
        """
        fills the mask array with ones if it isn't there yet
        """
        if type(self.mask) == bool:
            self.mask = np.ones_like(self.data)
        
    def build_static_mask(self,mask,points = False):
        """
        Makes a satic 2d mask for all data
        or only the time points listed in points option
        mask is 1.0 for good, np.nan for bad
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
                    mask = np.ones([self.n_t,self.m,self.n])
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n])
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n])
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
            temp_array = np.nanmean([self.data[j,:,:]*mask[j] for i in idx for j in i if j>=t0 and j<=tE],axis = 0)
            return temp_array

    def clim_mean(self,mask = False,year_set = [],time_set = []):
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
            temp_array = np.empty([self.periods])
            for mn in range(self.periods):
                idx = self.yrpd[y0:yE+1,mn].compressed()
                temp_array[mn] = np.nanmean([self.data[i]*mask[i] for i in idx if i>=t0 and i<=tE])
            return temp_array


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
        
        [n_t,m,n] = np.shape(data)
        self.n_t = n_t
        self.m = m
        self.n = n
        self.nyrs = dates[-1].year - dates[0].year + 1 
        self.yrpd = np.ma.empty([self.nyrs,periods],dtype = int)
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
                mt = int(self.dates[tt].days*periods/365) - 1
    #             print(yr,mt,tt)
                self.yrpd[yr,mt] = tt
        self.yrpd.mask = self.yrpd < 0
        self.files = True

        
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
                    mask = np.ones([self.n_t,self.m,self.n])
                elif mask:
                    mask = self.mask
                else:
                    mask = np.ones([self.n_t,self.m,self.n])
            elif (np.shape(mask)[0] != self.m 
                |np.shape(mask)[1] != self.n):# check mask dimension)
                print("Mask array incorrect shape, ignoring it")
                mask = np.ones([self.n_t,self.m,self.n])
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
            if magnitude:
                temp_x = np.nanmean(
                [np.hypot(self.x[j],self.y[j])*mask[j] for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                return temp_x
            else:
                temp_x = np.nanmean(
                    [self.x[j]*mask[j] for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                temp_y = np.nanmean(
                    [self.y[j]*mask[j] for i in idx for j in i if j>=t0 and j<=tE],
                            axis = 0)
                return temp_x,temp_y


    def clim_mean(self,mask = False,magnitude = False,year_set = [],time_set = []):
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
            elif time_set[0]>self.nyrs:
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
                    temp_x[mn] = np.nanmean([np.hypot(self.x[i],self.y[i])*mask[i] for i in idx if i>=t0 and i<=tE])
                return temp_x
            else:
                temp_x[mn] = np.nanmean(
                    [self.x[i]*mask[i] for i in idx if i>=t0 and i<=tE])
                temp_y[mn] = np.nanmean(
                    [self.y[i]*mask[i] for i in idx if i>=t0 and i<=tE])
                return temp_x,temp_y

    def save(filename):
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
        DY_out.yrpd[:,:] = temp_yrpd[:,:]
        DY_out.yrpd.mask = DY_out.yrpd < 0
#         DY.n_t = npzfile["DY_n_t"]  
#         self.m = npzfile["DY_m"] = m
#         self.n = npzfile["DY_n"] = n
        DY_out.saved = True
        return DY_out
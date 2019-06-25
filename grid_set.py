# here is the class that holds all the data days/months
# it has all the gridding scripts needed
# it will save load all the data/days/months as needed

import numpy as np
import pandas as pd
import datetime
import shutil
import os
from invoke import run
from netCDF4 import Dataset
from numba import jit
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import data_year as dy
from dateutil.relativedelta import relativedelta
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata

class grid_set:
# will make one of these at a time point (as a datetime) defined by timestart

    def __init__(self,mplot):
        self.mplot = mplot
        self.proj = True
        self.files = False
        self.saved = False
        self.grid = False
        self.gridinfo = False
        self.masked = False
        self.data = False
  
        
#     def import_regrid
#     # takes another data, on a seperate lon/lat, and regrids into a data/day/month
    

#     def import_regrid_nc
        
#     def import_regrid_vec
#     # takes vector input, a seperate lon/lat, and regrids into a data/day/month
#     # makes use of rotate/regrid/rotate methods to get it all pretty

#     def import_regrid_vec_nc
        


    def set_proj(self,mplot):
         # puts in a projection mplot too
#        # make make sure you keep hold of regridding projs
        self.mplot = mplot
        self.proj = True

    def set_grid_dxy(self,dxRes,dyRes):
       # creates a grid depending on wanted resolution 
        if self.proj:
            nx = int((self.mplot.xmax-self.mplot.xmin)/dxRes)+1
            ny = int((self.mplot.ymax-self.mplot.ymin)/dyRes)+1
            lons, lats, xpts, ypts = self.mplot.makegrid(nx, ny, returnxy=True)
            self.lons = lons
            self.lats = lats
            self.xpts = xpts
            self.ypts = ypts
            self.dxRes = dxRes
            self.dyRes = dyRes
            self.grid = True
            self.m = nx
            self.n = ny
            print("Got a grid res = ",nx," x ",ny)
            print("Note that all grid info is in ny x nx grids, whilst data is in nx x ny")
        else: print("Projection not defined yet, do that first")

    def set_grid_mn(self,nx,ny):
       # creates a grid depending on wanted no. of points 
        if self.proj:
            lons, lats, xpts, ypts = self.mplot.makegrid(nx, ny, returnxy=True)
            self.lons = lons
            self.lats = lats
            self.xpts = xpts
            self.ypts = ypts
            self.grid = True
            self.dxRes = (self.mplot.xmax-self.mplot.xmin)/(nx - 1)
            self.dyRes = (self.mplot.ymax-self.mplot.ymin)/(ny - 1)
            self.m = nx
            self.n = ny
            print("Got a grid res = ",nx," x ",ny)
        else: print("Projection not defined yet, do that first")

    def get_grid_info(self):
       # creates a grid depending on wanted no. of points 
        # print( self.grid and (not self.gridinfo))
        if self.grid and (not self.gridinfo):
            #iterate over the grid to get dimensions and angles
            # first iterate all x dimensions - m-1/n array
            # then  iterate all y dimensions - m/n-1 array
            xdims = np.empty([self.n,self.m-1])
            ydims = np.empty([self.n-1,self.m])
            self.xdist = np.empty([self.n,self.m])
            self.ydist = np.empty([self.n,self.m])
            self.ang_c = np.empty([self.n,self.m])
            self.ang_s = np.empty([self.n,self.m])
            for i in range(self.m-1):
                 for j in range(self.n):
                     xdims[j,i] = ellipsoidal_distance(
                         self.lons[j,i],self.lats[j,i],
                         self.lons[j,i+1],self.lats[j,i+1],deg=True)
            for i in range(self.m):
                 for j in range(self.n-1):
                     ydims[j,i] = ellipsoidal_distance(
                         self.lons[j,i],self.lats[j,i],
                         self.lons[j+1,i],self.lats[j+1,i],deg=True)

            # then average the available distances i-1,i j-1,j
            for i in range(self.m):
                 for j in range(self.n):
                     self.xdist[j,i] = np.nanmean(xdims[j,:i+1][-2:])
                     self.ydist[j,i] = np.nanmean(ydims[:j+1,i][-2:])
            print("Grid distances calculated: ",np.nanmean(self.xdist)," x ",np.nanmean(self.ydist))
                     
            # then  iterate all angles - this is all points plus the extra possible angles
            # pad the lon lat arrays for iteration
            lon_pad = np.pad(self.lons, (1,1), 'linear_ramp', end_values=(np.nan))
            lat_pad = np.pad(self.lats, (1,1), 'linear_ramp', end_values=(np.nan))
            for i in range(self.m):
                 for j in range(self.n):
                     # i + angle
                     xPlus_c,xPlus_s = lon_lat_angle(lon_pad[j+1,i+1],lat_pad[j+1,i+1],
                                            lon_pad[j+1,i+2],lat_pad[j+1,i+2],return_trig = True,deg=True)
                     xMins_c,xMins_s = lon_lat_angle(lon_pad[j+1,i+1],lat_pad[j+1,i+1],
                                            lon_pad[j+1,i],lat_pad[j+1,i],return_trig = True,deg=True)
                     yPlus_c,yPlus_s = lon_lat_angle(lon_pad[j+1,i+1],lat_pad[j+1,i+1],
                                            lon_pad[j+2,i+1],lat_pad[j+2,i+1],return_trig = True,deg=True)
                     yMins_c,yMins_s = lon_lat_angle(lon_pad[j+1,i+1],lat_pad[j+1,i+1],
                                            lon_pad[j,i+1],lat_pad[j,i+1],return_trig = True,deg=True)
                     # average all the components first checking the orientation
                     # if j == 20 and i ==12:
                         # print([xPlus_c,xMins_c,yPlus_c,yMins_c])
                         # print([xPlus_s,xMins_s,yPlus_s,yMins_s])
                     self.ang_c[j,i] = np.nanmean([-xPlus_s, xMins_s, yPlus_c,-yMins_c])
                     self.ang_s[j,i] = np.nanmean([ xPlus_c, xMins_c, yPlus_s,-yMins_s])
            self.gridinfo = True
        else: print("Grid not defined yet, do that first")

    def save_grid(self,file):
        if self.grid and self.gridinfo:
            # save lat/lon pts 
            np.savez(file,
                lats = self.lats,
                lons = self.lons,
                xpts = self.xpts,
                ypts = self.ypts,
                dxRes = self.dxRes,
                dyRes = self.dyRes,
                m = self.m,
                n = self.n,
                ang_c = self.ang_c,
                ang_s = self.ang_s,
                xdist = self.xdist,
                ydist = self.ydist)
            print("Grid saved in "+file)
        else:
            print("No grid to save - run get_grid_info")

    def load_grid(self,file):
        npzfile =  np.load(file)
        self.lats = npzfile["lats"]
        self.lons = npzfile["lons"]
        self.xpts = npzfile["xpts"]
        self.ypts = npzfile["ypts"]
        self.dxRes = npzfile["dxRes"] 
        self.dyRes = npzfile["dyRes"] 
        self.m = npzfile["m"] 
        self.n = npzfile["n"] 
        self.ang_c = npzfile["ang_c"] 
        self.ang_s = npzfile["ang_s"] 
        self.xdist = npzfile["xdist"] 
        self.ydist = npzfile["ydist"] 
        self.grid = True
        self.gridinfo = True
        print("Loaded a grid: "+file)

    def check_grid(self):
        # makes sure the projection and loaded grid are consistent
        if self.proj and self.grid and self.gridinfo:
            proj_dim = self.mplot.xmax - self.mplot.xmin
            proj_dim = proj_dim/self.m
            print("Projection av xdim = ",proj_dim)
            print("dxRes              = ",self.dxRes)
            print("xdist av           = ",np.mean(self.xdist))


    def get_grid_mask(self,inflate = 0.0):
        # makes a land mask for each point then inflates by a distance m
        # makes a land mask for each point then inflates by a distance m
        if self.masked:
            print("Already masked, do it again? set mask = False first")
        else:
            self.mask = np.ones([self.m,self.n])
            for i in range(self.m):
                for j in range(self.n):
                    if self.mplot.is_land(self.xpts[j,i],self.ypts[j,i]):
                         self.mask[i,j] = np.nan
            inf_mask = np.ones([self.m,self.n])
            if (inflate>0.0) and self.gridinfo:
                self.mask_inflate = inflate
                for i in range(self.m):
                    for j in range(self.n):
                        if np.isnan(self.mask[i,j]):
                            inf_p = int(inflate/np.hypot(self.xdist[j,i],self.ydist[j,i]))
                            inf_mask[i-inf_p:i+inf_p+1,j-inf_p:j+inf_p+1] = np.nan
                self.mask = inf_mask
        self.masked = True

    def save_mask(self,file):
        if self.masked:
            # save lat/lon pts 
            np.savez(file,
                mask = self.mask,
                mask_inflate = self.mask_inflate,
                m = self.m,
                n = self.n)
            print("Mask saved in "+file)
        else:
            print("No mask to save - run get_grid_mask")


    def load_mask(self,file):
        if self.masked:
            print("Masked already!")
        elif self.gridinfo:
            # save lat/lon pts 
            npzfile =  np.load(file)
            self.mask = npzfile["mask"]
            self.mask_inflate = npzfile["mask_inflate"]
            m_check = npzfile["m"] 
            n_check = npzfile["n"] 
            if (m_check == self.m)&(n_check == self.n):
                print("Loaded mask, ",m_check," x ",n_check," inflated by ",self.mask_inflate)
                self.masked = True
            else: 
                print("Gird and mask dimensins inconsistent, check them") 
                print("Mask",m_check," x ",n_check," Grid, ",self.m," x ",self.n)

def read_nc_single(ncfile,grid_set,lonlatk,valk,fill_lonlat = False):
    """
    # read and grids, then regrids a single data slice netcdf
    # data array
    # slightly flexible,
    # lonlatk = ['x','y'], say, gives the variable names for lon/lat
    # valkk = ['data'] gives the variable name for the data you want 
    # fill_lonlat = True, allows the code to deal with an even lon/lat grid
    # where the lon lat dimensions are given by a single dimension array
    """
    data_nc = Dataset(ncfile)
    lons = data_nc.variables[lonlatk[0]][:]
    lats = data_nc.variables[lonlatk[1]][:]
    d_array = data_nc.variables[valk[0]][:]
    # if the lat_lons need filling - do it
    if fill_lonlat:
        lon_a,lat_a = np.meshgrid(lons,lats)
    else:
        lon_a = lons
        lat_a = lats
    # regrid depending upon m and grid
    x_nc, y_nc = grid_set.mplot(lon_a.data, lat_a.data)
    new_d_array = griddata((x_nc[~d_array.mask].ravel(), y_nc[~d_array.mask].ravel()),
                d_array[~d_array.mask].ravel(), (grid_set.xpts, grid_set.ypts),
                method='linear')
    return new_d_array


def geo_gradient(array,grid_set):
    """
    gradient function that will take the grid info from the 
    grid_set type class to get gradients 
    the array has to be consistent with the grid set class so it can access the x/ydist parameters
    """
    # check if grid_set has grid info
    if not grid_set.gridinfo:
        print("No grid_set geo grid info - no result")
        return False
    in_mn = np.shape(array)
    if in_mn[0]!=grid_set.m or in_mn[1]!=grid_set.n :
        print("input array or geo grid_set not consistently shaped")
        return False
    else:
        out_Dax = np.empty_like(array) 
        out_Day = np.empty_like(array) 
        # np gradient can't do an uneven array
        # so we iterate along the columns, then the rows 
        # taking the gradient each time
        # 1 . columns
        for i in range(grid_set.m):
            temp_space = [np.sum(grid_set.ydist[i,0:j+1]) for j in range(grid_set.n)]
            out_Day[i,:] = np.gradient(
            array[i,:],temp_space)
        # 2 . rows
        for j in range(grid_set.n):
            temp_space = [np.sum(grid_set.xdist[0:i+1,j]) for i in range(grid_set.m)]
            out_Dax[:,j] = np.gradient(
            array[:,j],temp_space)
        return out_Dax,out_Day
    
def de_ripple(array1,array2,rip_filt_std = 1,filt_ring_sig = 5,force_zero = False):
    # find the ripples by subtracting the arrays
    ripples = array1 - array2
    # fast fourier transform the difference
    rip_spec  = np.fft.fft2(np.double(ripples))
    rip_spec2 = np.fft.fftshift(rip_spec)
    # find the ring sprectrum the contains the ripples
    filt_ring = np.ones_like(array1)
    spec_r = np.mean(rip_spec2) + rip_filt_std*np.std(rip_spec2)
    filt_ring[rip_spec2>spec_r] = 0.0
    filt_ring = gaussian_filter(filt_ring,sigma = filt_ring_sig)
    if not type(force_zero) == bool:
        filt_ring[rip_spec2>spec_r] = filt_ring[rip_spec2>spec_r]*force_zero  
    # use this filter ring to remove the array1 fft spectrum
    a1_spec  = np.fft.fft2(np.double(array1))
    a1_spec2 = np.fft.fftshift(a1_spec)
    a1_spec2 = a1_spec2*filt_ring
    back = np.real(np.fft.ifft2(np.fft.ifftshift(a1_spec2)))
    return back

def geo_filter(array,grid_set,distance,mask = False):
    """
    filter function that will take the grid info from the 
    grid_set type class to get filter distances
    the array has to be consistent with the grid set class so it can access the x/ydist parameters
    """
    # takes the DOT and filters out the geoid harmonics
    # hopefully can implement variable gradient using 
    # grid info
    # can dx/dyres if needed
    # check if grid_set has grid info
    if type(mask)==bool:
        mask = np.ones_like(array)
    elif (np.shape(mask)[0] != grid_set.m 
           |np.shape(mask)[1] != grid_set.n):# check mask dimension)
        print("Mask array incorrect shape, ignoring it")
        mask = np.ones([grid_set.m,grid_set.n])
    if not grid_set.gridinfo:
        print("No grid_set geo grid info - no result")
        return False
    in_mn = np.shape(array)
    if in_mn[0]!=grid_set.m or in_mn[1]!=grid_set.n :
        print("input array or geo grid_set not consistently shaped")
        return False
    else:
        V = np.empty_like(array) 
        W = np.empty_like(array) 
        out_array = np.empty_like(array) 
        f_sig =[distance/d for d in [grid_set.dxRes,grid_set.dyRes]] # some function of the radius given..
        V[:,:]=array*mask
        V[np.isnan(V)]=0
        VV=gaussian_filter(V,sigma=f_sig)

        W[:,:]=0*array+1
        W = W*mask
        W[np.isnan(W)]=0
        WW=gaussian_filter(W,sigma=f_sig)

        out_array[:,:]=VV/WW
        out_array[np.isnan(array)] = np.nan
        
        return out_array

# takes generic data and regrids it into a data_year
def regrid_data(data,dates,lons,lats,grid_set,periods,
                fill_lonlat = False):
    """
    makes a data year object, nicely regridded on D_Class grid
    time dimension of data is default 0 
    currently setup to access list of lists, or arrays
    first list access is the time point
    retains old netcdf option to fill lat lon arrays from singular 
    axis arrays
    otherwise lon/lat need to be of the same shape as the data time slice
    periods is the number of time slices per year, ie. 12 for monthlies
    """
    n_t = np.shape(data)[0]
    
    new_d_array = np.empty([n_t,grid_set.m,grid_set.n])
    # if the lat_lons need filling - do it
    if fill_lonlat:
        lon_a,lat_a = np.meshgrid(lons,lats)
    else:
        lon_a = lons
        lat_a = lats
    # regrid depending upon m and grid
    x_d, y_d = grid_set.mplot(lon_a, lat_a)
    for tt in range(n_t):
        new_d_array[tt,:,:] = griddata((x_d.ravel(), y_d.ravel()),
                data[tt][:].ravel(), (grid_set.xpts.T, grid_set.ypts.T),
                method='linear')
    return dy.data_year(new_d_array,dates,periods)

# takes generic data and regrids it into a data_year
def regrid_vectors(x,y,dates,lons,lats,grid_set,periods,
                fill_lonlat = False,vector_angles = False):
    """
    makes a vector data year object, nicely regridded on D_Class grid
    time dimension of data is default 0 
    currently setup to access list of lists, or arrays
    first list access is the time point
    retains old netcdf option to fill lat lon arrays from singular 
    axis arrays
    otherwise lon/lat need to be of the same shape as the data time slice
    periods is the number of time slices per year, ie. 12 for monthlies
    # the original vectors may need to be rotated back to be square to
    # lon lat so they can be regridded
    # if vector_angles = false then they are already square ie on an x/y frame sqaure to lon/lat
    # otherwise vector_angles is the same shape as lon/lats etc 
    #  and is angle positive from gridded data x/y to lon/lat  
    # ie positive rotational angle from local y positive dimension to true north
    # so angle array is consistent to gridinfo method on a grid_set - so you can use that.
    """
    n_t = np.shape(x)[0]
    
    new_x_array = np.empty([n_t,grid_set.m,grid_set.n])
    new_y_array = np.empty([n_t,grid_set.m,grid_set.n])
    # if the lat_lons need filling - do it
    if fill_lonlat:
        lon_a,lat_a = np.meshgrid(lons,lats)
    else:
        lon_a = lons
        lat_a = lats
    if type(vector_angles) == bool:
        orig_c = np.ones_like(lon_a)
        orig_s = np.zeros_like(lon_a)
    else: 
        orig_c = np.cos(np.deg2rad(vector_angles))
        orig_s = np.sin(np.deg2rad(vector_angles))
    # regrid depending upon mplot and grid
    x_d, y_d = grid_set.mplot(lon_a, lat_a)
    for tt in range(n_t):
        # rotating back to lon lat
        orig_x = x*orig_c - y*orig_s
        orig_y = y*orig_c + x*orig_s
        # regridding
        temp_x = griddata((x_d.ravel(), y_d.ravel()),
                orig_x.ravel(), (grid_set.xpts.T, grid_set.ypts.T),
                method='linear')
        temp_y = griddata((x_d.ravel(), y_d.ravel()),
                orig_y.ravel(), (grid_set.xpts.T, grid_set.ypts.T),
                method='linear')
        
        # rotating to the new grid
        new_x_array[tt] = temp_x*grid_set.ang_c + temp_y*grid_set.ang_s 
        new_y_array[tt] = temp_y*grid_set.ang_c - temp_x*grid_set.ang_s 
        
    return dy.vec_data_year(new_x_array,new_y_array,dates,periods)


@jit
def ellipsoidal_distance(long1, lat1, long2, lat2,deg=False,eps=1e-10):
    """
    (long1, lat1, long2, lat2) all in radians
    outputs a distance in m
    """
    if deg:
        long1 = np.deg2rad(long1)
        lat1  = np.deg2rad(lat1)
        long2 = np.deg2rad(long2)
        lat2  = np.deg2rad(lat2)

    a = 6378137.0 # equatorial radius in meters
    f = 1/298.257223563 # ellipsoid flattening
    b = (1 - f)*a
    tolerance = eps # to stop iteration

    phi1, phi2 = lat1, lat2
    U1 = np.arctan((1-f)*np.tan(phi1))
    U2 = np.arctan((1-f)*np.tan(phi2))
    L1, L2 = long1, long2
    L = L2 - L1
    i = 0

    lambda_old = L + 0

    while True:

        t =  (np.cos(U2)*np.sin(lambda_old))**2
        t += (np.cos(U1)*np.sin(U2) - np.sin(U1)*np.cos(U2)*np.cos(lambda_old))**2
        sin_sigma = t**0.5
        cos_sigma = np.sin(U1)*np.sin(U2) + np.cos(U1)*np.cos(U2)*np.cos(lambda_old)
        sigma     = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha    = np.cos(U1)*np.cos(U2)*np.sin(lambda_old) / (sin_sigma)
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = cos_sigma - 2*np.sin(U1)*np.sin(U2)/(cos_sq_alpha+1e-12)
        C            = f*cos_sq_alpha*(4 + f*(4-3*cos_sq_alpha))/16

        t          = sigma + C*sin_sigma*(cos_2sigma_m + C*cos_sigma*(-1 + 2*cos_2sigma_m**2))
        lambda_new = L + (1 - C)*f*sin_alpha*t
        if np.abs(lambda_new - lambda_old) <= tolerance:
            break
        elif i > 1000:
            return np.nan
            break
        else:
            lambda_old = lambda_new
            i += 1

    u2 = cos_sq_alpha*((a**2 - b**2)/b**2)
    A  = 1 + (u2/16384)*(4096 + u2*(-768+u2*(320 - 175*u2)))
    B  = (u2/1024)*(256 + u2*(-128 + u2*(74 - 47*u2)))
    t  = cos_2sigma_m + 0.25*B*(cos_sigma*(-1 + 2*cos_2sigma_m**2))
    t -= (B/6)*cos_2sigma_m*(-3 + 4*sin_sigma**2)*(-3 + 4*cos_2sigma_m**2)
    delta_sigma = B * sin_sigma * t
    s = b*A*(sigma - delta_sigma)

    return s


@jit
def lon_lat_angle( lon1,lat1,lon2,lat2,deg=False,return_trig = False ):
    """
    #LAT_LON_ANGLE finds the geodesic angle from point 1 to point 2 
    #(lat lon in radians)
    #   This done by a series of axes rotations to get the 1st point at 0,0
    #   keeping the second the same relative distance apart. The roataion
    #   needed to get the 2nd point directly north of the 1st is the geodesic
    #   angle.
    """
    if deg:
        lon1 = np.deg2rad(lon1)
        lat1 = np.deg2rad(lat1)
        lon2 = np.deg2rad(lon2)
        lat2 = np.deg2rad(lat2)
    
    C_lat=np.cos(lat2);
    S_lat=np.sin(lat2);
    C_lon=np.cos(lon2);
    S_lon=np.sin(lon2);
    
    C_1=np.cos(-lon1);
    S_1=np.sin(-lon1);
    C_2=np.cos(-lat1);
    S_2=np.sin(-lat1);
    
    A1=[[C_1, -S_1, 0],
        [S_1,  C_1, 0],
        [0,    0,   1]]
    
    A2=[[C_2, 0, -S_2],
        [0,   1,  0  ],
        [S_2, 0,  C_2]]
    
    Borig=[C_lat*C_lon,
           C_lat*S_lon,
           S_lat      ];
    
    B=np.matmul(A2,A1)
    B=np.matmul(B,Borig)
    # print(B)
    
    if return_trig:
        scale=np.hypot(B[1],B[2])
        
        angle_sin=-B[1]/scale
        angle_cos= B[2]/scale

        return angle_cos, angle_sin
    else:
        angle=np.arctan2(-B[1],B[2])
        
        return angle

    
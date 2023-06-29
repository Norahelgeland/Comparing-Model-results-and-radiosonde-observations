"""
author: Nora Helgeland
date: May, 2023

"""

import numpy as np
import pandas as pd
from datetime import datetime
import time
import csv
import readline
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import os
import math
import sys
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import xarray
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gc
from numpy import datetime64
import fimex_xarray
from netCDF4 import Dataset
import time
from radiosonde_class2 import Radiosonde_csv
from tqdm import tqdm
import os

gc.collect()

def BilinnerarPoints(ds, lat, lon):


    ds_lat = ds['latitude'].values
    ds_lon = ds['longitude'].values
        
    #find distance between all the latitudes and longitudes in the dataset and the observation latitude
    X=np.sqrt(np.square(ds_lat-lat) + np.square(ds_lon-lon))
    idx=np.where(X == X.min())

    ix=idx[1] # longitude with smallest distance
    iy=idx[0] # latitude with smallest distance
    #print('ix',ix)
    #print('iy',iy)
        
    #Find nearest neighbors y,x for the latitude 
    iy1=iy
    iy2=iy+1
    if ((ds_lat[iy,ix]-lat) > 0):
        iy2=iy
        iy1=iy-1
    #
    ix1=ix
    ix2=ix+1
    if ((ds_lon[iy,ix]-lon) > 0):
        ix2=ix
        ix1=ix-1

        #Bilinear interpolation coefficients s and t
    s = 1


    if ((ds_lat[iy2,ix1] - ds_lat[iy1,ix1]) > 0):
        s = (lat - ds_lat[iy1,ix]) / (ds_lat[iy2,ix] - ds_lat[iy1,ix])

        
    t = 1
    if ((ds_lon[iy1,ix2] - ds_lon[iy1,ix1]) > 0):
        t = (lon - ds_lon[iy,ix1]) / (ds_lon[iy,ix2] - ds_lon[iy,ix1])

    
    return s,t,ix1,ix2,iy1,iy2 


def Nearest_xy_point(ds, lat, lon):


    ds_lat = ds['latitude'].values
    ds_lon = ds['longitude'].values
        
    #find distance between all the latitudes and longitudes in the dataset and the observation latitude
    X=np.sqrt(np.square(ds_lat-lat) + np.square(ds_lon-lon))
    idx=np.where(X == X.min())

    ix=idx[1] # longitude with smallest distance
    iy=idx[0] # latitude with smallest distance


    return ix, iy 



def time_interpolate_points(ds, time1):
    
    """
    Innput:
    File or array?
    
    Ourtput:
    A linnear interpolation weight t, and the two corresponding time points
    
    """
    timepoints = ds["time"].values

    k = min(timepoints, key=lambda x: abs(x - time1))

    # Selecting the time point with minimum distance
    it=np.where(timepoints == k)
    it = int(it[0])
    
    t1=it
    t2=it+1
    
    if ((timepoints[it]-time1) > 0):
        t2=it
        t1=it-1
        
    t = 1
    if ((timepoints[t2] - timepoints[t1]) > 0):
        t = (time1 - timepoints[t1]) / (timepoints[t2] - timepoints[t1])
    

    return t,t1,t2 


def Bilinnear_interpolate(s, t, h1, h2, h3, h4):

    v1 = (1-s)*h1 + s*h2
    v2 = (1-s)*h3 + s*h4

    #Interpolate along longitude
    new = (1-t)*v1 + t*v2

    return new

def exner_levs(ds, time1, ix, iy):
    
   
    ds = ds.isel(time=time1)
    ds = ds.isel(y=iy, x=ix)

    #To find the presssure in the halflayer we need ap and bp which are constants representing the full layers

    ap = xarray.DataArray(ds["ap"].values, dims=["kh"])
    bp = xarray.DataArray(ds["b"].values, dims=["kh"])

    p_full = ap+bp*ds["surface_air_pressure"]
    p_full=p_full.squeeze()
   
    exner =  np.power(p_full/(ds["surface_air_pressure"][0]), 0.286)
    
    return exner


def interpolate_exner(ds, exn_lev, exner, variable, time1, ix, iy):


    ds = ds.isel(time=time1)

    var_list=ds[variable].isel(y=iy, x=ix).values
    #var_list = np.append(var_list, ds[variable_ground].isel(y=iy,x=ix,height33=0).values[0]) 

    # Finds min distance
    ih = np.argmin((exner - exn_lev)**2)

    # Selecting the time point with minimum distance
    ih = int(ih)

    h1=ih
    h2=ih+1


    if ((exner[ih]-exn_lev) > 0):
        h2=ih
        h1=ih-1
        #print("down")
        
    h=1
    if ((exner[h2] - exner[h1]) > 0):
        h = (exn_lev - exner[h1]) / (exner[h2] - exner[h1])
        #print("here")
        #print("h")


    variable_1= var_list[h1]
    variable_2= var_list[h2]

   
    new_variable = variable_1+(variable_2-variable_1)*h


    return new_variable
      


def interpolate_4dims(ds, sonde_object, variable):

    """
    Interpolation in time

    output:
    The interpolated variables
    """

    new_var = []
  
    time_init=sonde_object.data["time"][0]
    ix, iy = Nearest_xy_point(ds, sonde_object.lat , sonde_object.lon)


    data = ds.isel(time=0, y=iy, x=ix)
    exner = exner_levs(ds, 0, ix, iy).values

    exner=exner.squeeze()
 
    sonde_object.make_exner_levs(data["surface_air_pressure"].values.squeeze())

    sonde_exner=sonde_object.data.loc[sonde_object.data["exner"]<exner[-1]]
    sonde_exner=sonde_exner.reset_index()

    for row,t in enumerate(sonde_object.time[0:len(sonde_exner["exner"])]):
  
        time_inter = time_interpolate_points(ds, t)
        
        exn_lev = sonde_exner["exner"][row]

        #Finding the height levels in both time points
        if time_inter[0]==1:

        
            h = interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], ix, iy)       
            new_var.append(h[0])
     
        else:
               
            h1 = interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], ix, iy)  
            h2 = interpolate_exner(ds, exn_lev, exner, variable, time_inter[2], ix, iy)
              
    
            #interpolate in time
            new = h1+(h2-h1)*time_inter[0]
          
            new_var.append(new[0])

       
    return new_var, sonde_exner["exner"]


if __name__=="__main__":

    #labelling the different jobs being sent to the big computer
    task_id = int(os.getenv("SGE_TASK_ID"))-1
    i_sonde = task_id

    # loading the Chernobyl data from the Arome model
    config = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/netcdf/cdmGribReaderConfigArome2_5.xml"
    
    arome_file = f"/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/gribml/deter_19860427_12.grbml"
    
    ds = xarray.open_dataset(arome_file, config=config, engine="fimex").isel(time=slice(12, -1))

    b = datetime64(f'1986-04-28T00:00:00.000000000')

    # Finding the wind direction
    file_alpha = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/alpha_chernobyl.nc"
 
    ds_alpha = xarray.open_dataset(file_alpha)   
    ds_alpha = ds_alpha["alpha"]
    

    file = sorted(os.listdir(f"/home/norah/master/Rs_data_bufr/1986042800"))[i_sonde]
    file_list=file.strip(".csv")
    file_list1=file_list.split("_")

    Radiosonde1 = Radiosonde_csv(f"/home/norah/master/Rs_data_bufr/1986042800/{file_list1[0]}_{file_list1[1]}.csv", b)
        
    tx, exner = interpolate_4dims(ds, Radiosonde1, "x_wind_ml")
    ty, exner = interpolate_4dims(ds, Radiosonde1, "y_wind_ml")

    x,y = Nearest_xy_point(ds, Radiosonde1.lat, Radiosonde1.lon)
 
    """
    Find the interpolated wind direction
    ---------------------------------------------------------------------------------------------------------
    """

    if x[0]<792:

        wdir = ds_alpha.isel(y=y[0], x=x[0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180

    else:

        wdir = -ds_alpha.isel(y=y[0], x=x[0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180


    wdir_data = xarray.DataArray(wdir.squeeze(), dims=["exner"], coords={"exner":exner})

    wdir_data = xarray.Dataset({"wdir_model": wdir_data})
  
    wdir_data.to_netcdf(f"/home/norah/master/data/trial_folder_bufr/wdir/2800_wdir/{file_list1[0]}_{file_list1[1]}_wdir_model.nc")


    """
    Find the interpolated wind speed
    ----------------------------------------------------------------------------------------------------------
    """

    w_speed = np.sqrt(np.square(ty)+np.square(tx))

    wspeed_data = xarray.DataArray(w_speed.squeeze(), dims=["exner"], coords={"exner":exner})

    wspeed_data = xarray.Dataset({"wspeed_model": wspeed_data})
  
    wspeed_data.to_netcdf(f"/home/norah/master/data/trial_folder_bufr/wspeed/2800_wspeed/{file_list1[0]}_{file_list1[1]}_wspeed_model.nc")
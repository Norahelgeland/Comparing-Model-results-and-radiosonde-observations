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

    """
    Innput
    -------------------------------------------------------------------------------------------------
    ds: xarray dataframe containing the lonitude and latitude and corresponding xy values 
    lat: target latitude(int)
    lon: target longitude (int)

    Output
    -------------------------------------------------------------------------------------------------
    The nearest grid points and the weight to preform bilinnear interpolation in the xy plane.
    """


    ds_lat = ds['latitude'].values
    ds_lon = ds['longitude'].values
        
    #find distance between all the latitudes and longitudes in the dataset and the observation latitude
    X1=np.sqrt(np.square(ds_lat-lat))
    X2=np.sqrt(np.square(ds_lon-lon))
    
    idx1=np.where(X1 == X1.min())
    idx2=np.where(X2 == X2.min())

    ix=idx2[0] # longitude with smallest distance
    iy=idx1[0] # latitude with smallest distance
    #print('ix',ix)
    #print('iy',iy)
        
    #Find nearest neighbors y,x for the latitude 
    iy1=iy
    iy2=iy+1
    if ((ds_lat[iy]-lat) > 0):
        iy2=iy
        iy1=iy-1
    #
    ix1=ix
    ix2=ix+1
    if ((ds_lon[ix]-lon) > 0):
        ix2=ix
        ix1=ix-1

        #Bilinear interpolation coefficients s and t
    s = 1


    if ((ds_lat[iy2] - ds_lat[iy1]) > 0):
        s = (lat - ds_lat[iy1]) / (ds_lat[iy2] - ds_lat[iy1])

        
    t = 1
    if ((ds_lon[ix2] - ds_lon[ix1]) > 0):
        t = (lon - ds_lon[ix1]) / (ds_lon[ix2] - ds_lon[ix1])


    return s,t,ix1,ix2,iy1,iy2 


def Nearest_xy_point(ds, lat, lon):

    """
    Innput
    -------------------------------------------------------------------------------------------------
    ds: xarray dataframe containing the lonitude and lkatitude and corresponding xy values 
    lat: target latitude(int)
    lon: target longitude (int)

    Output
    -------------------------------------------------------------------------------------------------
    The nearest grid point
    """

    ds_lat = ds['latitude'].values
    ds_lon = ds['longitude'].values

    #find distance between all the latitudes and longitudes in the dataset and the observation latitude
    X1=np.sqrt(np.square(ds_lat-lat))
    X2=np.sqrt(np.square(ds_lon-lon))

    idx1=np.where(X1 == X1.min())
    idx2=np.where(X2 == X2.min())
    ix=idx2[0] # longitude with smallest distance
    iy=idx1[0] # latitude with smallest distance
  

    return ix, iy 



def time_interpolate_points(ds, time1):
    
    """
    Innput
    --------------------------------------------------------------------------------------
    ds: xarray dataframe
    time1: target time. need to be in datetime64 format
    
    Ourtput
    --------------------------------------------------------------------------------------
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

    """
    Innput
    --------------------------------------------------------------------------------------
    interpolation weights and the four nearest xy points
    
    Ourtput
    --------------------------------------------------------------------------------------
    the interpolated variable

    """

    v1 = (1-s)*h1 + s*h2
    v2 = (1-s)*h3 + s*h4

    #Interpolate along longitude
    new = (1-t)*v1 + t*v2

    return new

def exner_levs(ds, time1, ix, iy):
    
    """
    Innput
    ------------------------------------------------------------------------------------------
    ds: xarray datset/array
    ix: position in x direction (int)
    iy: position in y direction (int)
    time1: index for position in time (int)
    
    Ourtput
    ------------------------------------------------------------------------------------------
    list containing model exner levels
    
    """

   
    ds = ds.isel(time=time1)
    ds = ds.isel(latitude=iy, longitude=ix, latitude1=iy, longitude1=ix)

    #To find the presssure in the halflayer we need ap and bp which are constants representing the full layers
  
    ap = xarray.DataArray(ds["ap0"].values, dims=["kh"])
    bp = xarray.DataArray(ds["b0"].values, dims=["kh"])

    p_full = ap+bp*np.exp(ds["surface_air_pressure"].values.squeeze())
    p_full=p_full.squeeze()
   
    exner =  np.power(p_full/(math.exp(ds["surface_air_pressure"].values.squeeze())), 0.286)
    
    return exner


def interpolate_exner(ds, exn_lev, exner, variable, time1, ix, iy):
    
    """
    Innput
    ------------------------------------------------------------------------------------------
    ds: xarray datset/array
    variable: the variable that is being interpolated (string)
    exn_lev: vertical position of sonde observation (int)
    exner: list of model exner levels
    ix: position in x direction (int)
    iy: position in y direction (int)
    time1: index for position in time (int)
    
    Ourtput
    ------------------------------------------------------------------------------------------
    A linnear interpolation weight h, and the two nearest height levels
    
    """

    ds = ds.isel(time=time1)

    var_list=ds[variable].isel(latitude1=iy, longitude1=ix).values
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
    Innput
    -----------------------------------------------------------------------------
    ds: xarray datset/array
    sonde_oject: containing the sounding data (pandas dataframe)
    variable: the variable that is being interpolated (string)

    output
    -------------------------------------------------------------------------------
    new_var: The exner(vertically) interpolated(not hosizontally) variables in a list
    new_var_hor_int: The interpolated(also interpolated horizontally)

    """

    new_var = []
  
    time_init=sonde_object.data["time"][0]
    ix, iy = Nearest_xy_point(ds, sonde_object.lat , sonde_object.lon)

    v = BilinnerarPoints(ds, sonde_object.lat, sonde_object.lon)

    data = ds.isel(time=0, latitude=iy, longitude=ix, latitude1=iy, longitude1=ix)

    #Model exner levels
    exner = exner_levs(ds, 0, ix, iy).values

    exner=exner.squeeze()

    #sonde_exner levels
    sonde_object.make_exner_levs(np.exp(data["surface_air_pressure"].values.squeeze()))

    sonde_exner=sonde_object.data.loc[sonde_object.data["exner"]<exner[-1]]
    sonde_exner=sonde_exner.reset_index()

    #vectors containing the height interpolated variables in the nearest xy grid points
    h1=[]
    h2=[]
    h3=[]
    h4=[]


    for row,t in enumerate(sonde_object.time[0:len(sonde_exner["exner"])]):
  
        #time_inter = time_interpolate_points(ds, t)
        time_inter = [1,0]
        
        exn_lev = sonde_exner["exner"][row]

        #Finding the height levels in both time points
        #if time_inter[0]==1: #For now it wil allways go in this loop since there is no temporal interpolation

        
        h1.append(interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], v[2], v[4])[0])
        h2.append(interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], v[2], v[5])[0])
        h3.append(interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], v[3], v[4])[0])
        h4.append(interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], v[3], v[5])[0])

            #new_var.append(h[0])
     
        #else:
               
        h = interpolate_exner(ds, exn_lev, exner, variable, time_inter[1], ix, iy)  
            #h2 = interpolate_exner(ds, exn_lev, exner, variable, time_inter[2], ix, iy)
              
    
            #interpolate in time
            #new = h1+(h2-h1)*time_inter[0]
          
        new_var.append(h[0])

    new_var_hor_int = np.squeeze(Bilinnear_interpolate(v[0], v[1], h1, h2, h3, h4))

    
    return new_var, new_var_hor_int sonde_exner["exner"]


if __name__=="__main__":

    #labelling the different jobs being sent to the big computer
    task_id = int(os.getenv("SGE_TASK_ID"))-1
    i_sonde = task_id

    # loading the Chernobyl data from the Arome model
    config = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/cdmGribReaderConfigEC_Era5.xml"
    
    arome_file = f"/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/ERA5/ma198605061212.mars"
    
    ds = xarray.open_dataset(arome_file, config=config, engine="fimex")#.isel(time=slice(12, -1))
    #ds = xarray.open_dataset(arome_file).isel(time=slice(12, -1))

    b = datetime64(f'1986-05-06T12:00:00.000000000')

    # Finding the wind direction
    file_alpha = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/alpha_chernobyl.nc"
 
    ds_alpha = xarray.open_dataset(file_alpha)   
    ds_alpha = ds_alpha["alpha"]
    

    file = sorted(os.listdir(f"/home/norah/master/Rs_data_bufr/1986050612"))[i_sonde]
    #file = "48.63_22.27.csv"

    file_list=file.strip(".csv")
    file_list1=file_list.split("_")

    Radiosonde1 = Radiosonde_csv(f"/home/norah/master/Rs_data_bufr/1986050612/{file_list1[0]}_{file_list1[1]}.csv", b)
        
    tx,tx_int, exner = interpolate_4dims(ds, Radiosonde1, "x_wind_ml")
    ty,ty_int, exner = interpolate_4dims(ds, Radiosonde1, "y_wind_ml")

    x,y = Nearest_xy_point(ds, Radiosonde1.lat, Radiosonde1.lon)
 
    """
    Find the interpolated wind direction. To also include horizontal interpolation swithc to tx_int ant ty_int
    ----------------------------------------------------------------------------------------------------------
    """

    #wdir = (np.arctan2(ty, tx)*180.0/math.pi+180)%360
    wdir = (90-np.arctan2(ty, tx)*180.0/math.pi+180)%360

    #if x[0]<279:

        #wdir = ds_alpha.isel(y=y[0], x=x[0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180

    #else:

        #wdir = -ds_alpha.isel(y=y[0], x=x[0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180


    wdir_data = xarray.DataArray(wdir.squeeze(), dims=["exner"], coords={"exner":exner})

    wdir_data = xarray.Dataset({"wdir_model": wdir_data})
  
    wdir_data.to_netcdf(f"/home/norah/master/data/trial_folder_era5/wdir/0612_wdir/{file_list1[0]}_{file_list1[1]}_wdir_model.nc")


    """
    Find the interpolated wind speed
    ----------------------------------------------------------------------------------------------------------
    """

    w_speed = np.sqrt(np.square(ty)+np.square(tx))

    wspeed_data = xarray.DataArray(w_speed.squeeze(), dims=["exner"], coords={"exner":exner})

    wspeed_data = xarray.Dataset({"wspeed_model": wspeed_data})
  
    wspeed_data.to_netcdf(f"/home/norah/master/data/trial_folder_era5/wspeed/0612_wspeed/{file_list1[0]}_{file_list1[1]}_wspeed_model.nc")
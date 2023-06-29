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
#from netCDF4 import Dataset, num2date, date2num
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
import time
from radiosonde_class import Radiosonde
from tqdm import tqdm


def hybrid_levs_to_height(ds, time):

    """
    This function uses hybrid coordinates to find the pressure half levels, and from there the geopotential height in each half layer.
    """

    ds = ds.isel(time=time)

    #To find the presssure in the halflayer we need ap and bp which are constants representing the full layers
    ap = ds["ap"].values
    bp = ds["b"].values

    # Lets move up the half layers to find ap1/2 and bp1/2
    ap_half = np.zeros(len(ap)+1) # Is zero at ground level
    bp_half = np.zeros(len(ap)+1)

    ap_half[0] = 0 
    bp_half [0] = 0 

    ap_half[-1] = 0 # Is zero at ground level
    bp_half[-1] = 1 # Is 1 at ground level so that the pressure is equal to the surface pressure

    p_surface = ds["surface_air_pressure"] 

    for k in range(len(ap), 0,-1):

        #print(ap_half[k])
        ap_half[k-1] = ap[k-1]*2-ap_half[k]
        bp_half[k-1] = bp[k-1]*2-bp_half[k]  
    

    ap_half = xarray.DataArray(ap_half, dims=["kh"])
    b_half = xarray.DataArray(bp_half, dims=["kh"])

    p_half = ap_half + b_half * p_surface
 
    #We need the virtual temperature to solve the hypsometric equation
    T = ds["air_temperature_ml"]
    qv = ds["specific_humidity_ml"]

    T_v = (1+0.61*qv)*T

    #The hypsometric equation
    geopot_levs = np.zeros_like(T)
   
    g=9.81
    height=np.zeros_like(T.isel(hybrid=0))
    R=287 #J/(K*kg)
     
    for k in range(len(ap_half)-2,2, -1):

        delta_z = R*T_v.isel(hybrid = k-1)/g*np.log(p_half.isel(kh = k-1)/p_half.isel(kh=k))
    
        height-=delta_z.squeeze()
        geopot_levs[k,:,:] = height

    geopot_levs = xarray.DataArray(geopot_levs[:, :, :], dims=["hybrid","y","x"], coords=[ds["hybrid"], ds["y"], ds["x"]])
  

    return geopot_levs



if __name__=="__main__":

    task_id = int(os.getenv("SGE_TASK_ID"))-1
    i_time=task_id
    sonde_time=pd.date_range("1986-04-26T00:00", freq="12h", periods=26)

    selected_time=sonde_time[i_time]
    arome_time= selected_time-np.timedelta64(12, "h")

    t=arome_time
     # loading the Chernobyl data from the Arome model
    config = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/netcdf/cdmGribReaderConfigArome2_5.xml"
    arome_file = f"/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/gribml/deter_1986{t.month:02}{t.day:02}_{t.hour:02}.grbml"

    ds = xarray.open_dataset(arome_file, config=config, engine="fimex").isel(time=slice(12, -1))
        #ds = ds.isel(time=5)
   
    geopot_levs = hybrid_levs_to_height(ds, time=0)
        
    ds2 = xarray.Dataset()
    ds2["geopot_levs"] = geopot_levs     
    
    #ds["time"].values)
    t=selected_time
    dir_name=f"geopot_levs/{t.day:02}{t.hour:02}"
    os.mkdir(dir_name) 
    ds2.to_netcdf(f"{dir_name}/geopot_levs_0.nc")

    for t in tqdm(range(1,len(ds["time"].values))):
        geopot = hybrid_levs_to_height(ds, time=t)
        #geopot_levs= xarray.concat([geopot_levs,geopot], dim="time")
        #geopot_levs = np.append(geopot_levs, geopot)

        ds2 = xarray.Dataset()
        ds2["geopot_levs"] = geopot_levs 

        ds2.to_netcdf(f"{dir_name}/geopot_levs_"+str(t)+".nc")

    

  



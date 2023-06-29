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
from netCDF4 import Dataset
import time
from radiosonde_class import Radiosonde
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


def dataSeriesLev(s,t,ix1,ix2,iy1,iy2,dataset, variable, time_step):

    """
    Innput:
    Billinnear interpolation weights s and t. Then four grid points and a dataset containing the variable

    Output:
    List with variable in each hybrid layer
    """
    start_time = time.time()
    variable_inter = dataset[variable]
    variable_inter = variable_inter.isel(time=time_step)
    #latitude[18, 60,].values
    #Move up all levels in one timestep
    #v = np.zeros(len(variable_inter))    
    #Get four corners to interpolate as lists
    ll = variable_inter.isel(y=iy1, x=ix1).values
    ul = variable_inter.isel(y=iy2, x=ix1).values
    lr = variable_inter.isel(y=iy1, x=ix2).values
    ur = variable_inter.isel(y=iy2, x=ix2).values

    #Interpolate along latitude
    v1 = (1-s)*ll + s*ul
    v2 = (1-s)*lr + s*ur

    #Interpolate along longitude
    yx = (1-t)*v1 + t*v2
    #print(yx)        
    #v = yx.reshape(yx.shape[0])
    end_time = time.time()
    print("Run time = {}".format(end_time-start_time))
    #gc.collect()
    
    return yx

def Bilinnear_interpolate(s, t, h1, h2, h3, h4):

    v1 = (1-s)*h1 + s*h2
    v2 = (1-s)*h3 + s*h4

    #Interpolate along longitude
    new = (1-t)*v1 + t*v2

    return new


def hybrid_levs_to_height(ds, time):

    """
    Height levels could be found here if many assumptions and simplifications are made. 
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

    p_surface = ds["surface_air_pressure"]#Also needs to be interpolated

    for k in range(len(ap), 0,-1):

        #print(ap_half[k])
        ap_half[k-1] = ap[k-1]*2-ap_half[k]
        bp_half[k-1] = bp[k-1]*2-bp_half[k]  
    
    #Prøve senere? 

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
     
    for k in range(len(ap_half)-2,1, -1):

        delta_z = R*T_v.isel(hybrid = k-1)/g*np.log(p_half.isel(kh = k-1)/p_half.isel(kh=k))
        #breakpoint()
        height-=delta_z.squeeze()
        geopot_levs[k,:,:] = height
        
    #breakpoint()
    geopot_levs = xarray.DataArray(geopot_levs, dims=["hybrid","y","x"], coords=[ds["hybrid"], ds["y"], ds["x"]])

    return geopot_levs


def height_interpolate_points(ds, geopot_levs, variable, variable_ground, height,x, y, time1):
    """
    Innput:
    xarray datset/array
    
    Ourtput:
    A linnear interpolation weight t, and the two corresponding time points
    
    """
 
    ds = ds.isel(time=time1)

    var_list=ds[variable].isel(y=y, x=x).values
    var_list = np.append(var_list, ds[variable_ground].isel(y=y,x=x,height33=0).values[0])
   

    geopot_levs = geopot_levs.isel(y=y,x=x).squeeze().values
    geopot_levs = geopot_levs[2:]
    
    #Adding the variable at 10 meters
    geopot_levs = np.append(geopot_levs,10)
    df_levs = pd.DataFrame(geopot_levs, columns = None)

    # Finds min distance
    ih = np.argmin((geopot_levs - height)**2)
    #print("height")
    #print(height)
 

    # Selecting the time point with minimum distance
    ih = int(ih)
  

    h1=ih
    h2=ih+1
    
    if ((geopot_levs[ih]-height) < 0):
        h2=ih
        h1=ih-1
        #print("down")
        
    h=1
    if ((geopot_levs[h2] - geopot_levs[h1]) < 0):
        h = (height - geopot_levs[h1]) / (geopot_levs[h2] - geopot_levs[h1])
        #print("here")
        #print("h")

    #print(geopot_levs[h1])
    #print(geopot_levs[h2])

    variable_1= var_list[h1]
    variable_2= var_list[h2]

    new_variable = variable_1+(variable_2-variable_1)*h
    
    #print(variable_1)
    #print(variable_2)


    #print("the new variable")
    #print(new_variable)

    return new_variable, h,h1,h2 
    
    


def interpolate_4dims(ds, sonde_object, variable, variable_ground, geopot_levs):

    """
    Here the interpolation takes place

    output:
    The interpolated variables
    """
    max_height=np.where(sonde_object.data["height(m)"]<7000)
    
    
    if len(max_height[0])<10:
        raise ValueError("Not enough data")

    limit = max_height[0].max()

    new_var = []
    height_list = []
    
    for row,t in enumerate(sonde_object.time):
  
        time_inter = time_interpolate_points(ds, t)

        
        h = sonde_object.data["height(m)"][row]

        height_list.append(h)
        v = BilinnerarPoints(ds, sonde_object.data["new_lat"][row], sonde_object.data["new_long"][row])
        #Finding the height levels in both time points
        if time_inter[0]==1:

            #ds = xarray.concat(ds["air_temperature_ml"].isel(time=time_inter[2]), ds["air_temperature_2m"].isel(time=time_inter[2]), dim="hybrid")
            #breakpoint()
            geopot_levs2 = geopot_levs.isel(time=time_inter[2])

            # Finding the four corners of the model cell
            h1= height_interpolate_points(ds, geopot_levs2, variable, variable_ground, h, v[2], v[4], time_inter[2])
      
            h2 = height_interpolate_points(ds, geopot_levs2, variable, variable_ground, h,v[2],v[5], time_inter[2])

            h3 = height_interpolate_points(ds, geopot_levs2, variable, variable_ground, h,v[3],v[4], time_inter[2])

            h4 = height_interpolate_points(ds, geopot_levs2, variable, variable_ground, h,v[3],v[5], time_inter[2])
       
            #Bilinnear interpolation in the xy plane
            t1 = Bilinnear_interpolate(v[0], v[1], h1[0], h2[0], h3[0], h4[0])
            new_var.append(t1[0])
            #print(t1)

        else:
            bi_inter=[]
            for i, geopot_levs1 in enumerate([geopot_levs.isel(time=time_inter[1]), geopot_levs.isel(time=time_inter[2])]):

                # Finding the four corners of the model cell
                h1= height_interpolate_points(ds, geopot_levs1, variable, variable_ground, h, v[2], v[4], time_inter[i+1])

                h2 = height_interpolate_points(ds, geopot_levs1, variable, variable_ground, h,v[2],v[5], time_inter[i+1])

                h3 = height_interpolate_points(ds, geopot_levs1, variable, variable_ground, h,v[3],v[4], time_inter[i+1])

                h4 = height_interpolate_points(ds, geopot_levs1, variable,variable_ground, h,v[3],v[5], time_inter[i+1])

                #Bilinnear interpolation in the xy plane
                bi_inter.append(Bilinnear_interpolate(v[0], v[1], h1[0], h2[0], h3[0], h4[0]))

    
            #interpolate in time
            new = bi_inter[0]+(bi_inter[1]-bi_inter[0])*time_inter[0]
            new_var.append(new[0])


    return new_var, height_list,limit


if __name__=="__main__":


    #Sending it to the supercomputer to save time
    i = int(os.getenv("SGE_TASK_ID"))-1
    #i_sonde = task_id%100

    #i_time=task_id//100
    #sonde_time=pd.date_range("1986-04-26T00:00", freq="12h", periods=26)

    #selected_time=sonde_time[i_time]
    #arome_time= selected_time-np.timedelta64(12, "h")
    
    # loading the Chernobyl data from the Arome model
    config = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/netcdf/cdmGribReaderConfigArome2_5.xml"
    #t=arome_time
    #arome_file = f"/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/gribml/deter_1986{t.month:02}{t.day:02}_{t.hour:02}.grbml"

    arome_file = f"/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/gribml/deter_19860425_12.grbml"

    file_alpha = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/alpha_chernobyl.nc"

    ds_alpha = xarray.open_dataset(file_alpha)   
    ds_alpha = ds_alpha["alpha"]

    ds = xarray.open_dataset(arome_file, config=config, engine="fimex").isel(time=slice(12, -1))

    #t=selected_time
    #b = datetime64(f'1986-{t.month:02}-{t.day:02}T{t.hour:02}:00:00.000000000')

    b = datetime64(f'1986-04-26T00:00:00.000000000')

    #geopot_levs = xarray.open_dataset("geopot_levs.nc")["geopot_levs"]
    #geopot_levs = xarray.open_mfdataset([f"geopot_levs/{t.day:02}{t.hour:02}/geopot_levs_"+str(i)+".nc" for i in range(5)], chunks={"time": 1}, concat_dim="time", combine="nested")["geopot_levs"]
    geopot_levs = xarray.open_mfdataset([f"geopot_levs/2600/geopot_levs_"+str(i)+".nc" for i in range(5)], chunks={"time": 1}, concat_dim="time", combine="nested")["geopot_levs"]
    #windir_data = xarray.open_mfdataset(["windir_data_"+str(i)+".nc" for i in range(19)], chunks={"time": 1}, concat_dim="time", combine="nested")["wdir"]

    # Finding the wind direction
    file_alpha = "/lustre/storeB/project/fou/kl/cerad/Meteorology/AROME-CHERNOBYL/alpha_chernobyl.nc"

    ds_alpha = xarray.open_dataset(file_alpha)   
    ds_alpha = ds_alpha["alpha"]
   
    rmse = []
    distance =[] 
    names = []
    inter_var= []

    #Interpolating all radiosondes at one timestep

    
    #file = sorted(os.listdir(f"RS_data/RS_{t.hour:02}_{t.day:02}_{t.month:02}_1986"))[i_sonde]
    file = sorted(os.listdir(f"RS_data/RS_00_26_04_1986"))[i]

    file_list1=file.strip(".txt")
    file_list=file_list1.split("_")
    names.append(file_list[0]+"_"+file_list[1])
    lat = float(file_list[2])
    long = float(file_list[3])

    #Radiosonde1 = Radiosonde(f"RS_data/RS_{t.hour:02}_{t.day:02}_{t.month:02}_1986/"+file, lat, long,b)
    Radiosonde1 = Radiosonde(f"RS_data/RS_00_26_04_1986/"+file, lat, long,b)
    Radiosonde1.find_data_no_horizontal_disp()


    tx,h,l= interpolate_4dims(ds, Radiosonde1, "x_wind_ml","x_wind_10m", geopot_levs)
    ty,h,l= interpolate_4dims(ds, Radiosonde1, "y_wind_ml", "y_wind_10m", geopot_levs)
    #temp,h,l= interpolate_4dims(ds, Radiosonde1, "temperature_ml", "temperature_2m", geopot_levs)

    v = BilinnerarPoints(ds, Radiosonde1.lat, Radiosonde1.lon)

    """
    Uncomment the lines below to find the interpolated wind direction
    ----------------------------------------------------------------------------------------------------------
    """

    if v[3][0]<792:

        wdir = ds_alpha.isel(y=v[5][0], x=v[3][0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180
        #x_correct = np.divide(tx,math.cos(ds_alpha.isel(x=v[3][0], y=v[5][0]).values))
        #y_correct = np.divide(ty,math.cos(ds_alpha.isel(x=v[3][0], y=v[5][0]).values))
    else:
        wdir = -ds_alpha.isel(y=v[5][0], x=v[3][0]).values+90-np.arctan2(ty, tx)*180.0/math.pi+180
        #x_correct = np.divide(tx,math.cos(-ds_alpha.isel(x=v[3][0], y=v[5][0]).values))
        #y_correct = np.divide(ty,math.cos(-ds_alpha.isel(x=v[3][0], y=v[5][0]).values))

    
    wdir_data = xarray.DataArray(wdir, dims=["height"], coords={"height":h})

    wdir_data = xarray.Dataset({"wdir_model": wdir_data})
  
    #wdir_data.to_netcdf(f"/home/norah/master/data/no_hdisp_wdir_int/int_sonde_data{t.hour:02}_{t.day:02}/{file_list1}_wdir_model.nc")

    wdir_data.to_netcdf(f"/home/norah/master/data/no_hdisp_wdir_int/int_sonde_data00_26/{file_list1}_wdir_model.nc")

    """
    Uncomment the four lines below to find the interpolated wind speed
    ----------------------------------------------------------------------------------------------------------
    """

    w_speed = np.sqrt(np.square(ty)+np.square(tx))

    wspeed_data = xarray.DataArray(w_speed, dims=["height"], coords={"height":h})

    wspeed_data = xarray.Dataset({"wspeed_model": wspeed_data})
  
    #wspeed_data.to_netcdf(f"/home/norah/master/data/no_hdisp_wspeed_int/int_sonde_data_wspeed{t.hour:02}_{t.day:02}/{file_list1}_wspeed_model.nc")
    
    wspeed_data.to_netcdf(f"/home/norah/master/data/no_hdisp_wspeed_int/int_sonde_data_wspeed00_26/{file_list1}_wspeed_model.nc")


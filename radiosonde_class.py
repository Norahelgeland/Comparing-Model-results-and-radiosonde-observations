"""
author: Nora Helgeland
date: May, 2023

"""


from __future__ import annotations
import pandas as pd
import math
import numpy as np
import geopy.distance
from datetime import datetime
from numpy import datetime64
import matplotlib.pyplot as plt

class Radiosonde:

    # class attribute

    
    # Instance attribute
    def __init__(self, filename, lat, lon, time):

        """
        Innput
        ------------------------------------------------------------------------
        filename is the name of the dataset containing information of a radiosonde
        (has to be a .txt file). 
        lat and long is the initial longitiude. 
        and latitude of the radiosonde.
        time: datetime64
        ------------------------------------------------------------------------

        """
        
        self.data =  pd.read_fwf(filename,header = None, skiprows = [0], names=["hPa", "height(m)", "Temp(C)", "DWPT", "Relh", "MIXR", "DRCT(deg)","WSPD(knot)", "8", "9", "10"])

        self.lat = lat
        self.lon = lon
        self.time = [time+np.timedelta64(int(height/5), 's') for height in self.data["height(m)"]]
        self.data["time"] = self.time
        self.distance: list[float] = [0]
        self.h_disp = 1
        self.data["WSPD(knot)"] = self.data["WSPD(knot)"][:]*0.514444444

    def find_horizontal_disp(self):


        #converting knots into meters per second
        self.data["DRCT(deg)"] = self.data["DRCT(deg)"].fillna(method='bfill')
        self.data["DRCT(deg)"] = self.data["DRCT(deg)"].fillna(method='ffill')
    
        self.data["height(m)"] = self.data["height(m)"].fillna(method='bfill')
        self.data["height(m)"] = self.data["height(m)"].fillna(method='ffill')

        self.data["WSPD(knot)"] = self.data["WSPD(knot)"].fillna(method='bfill')
        self.data["WSPD(knot)"] = self.data["WSPD(knot)"].fillna(method='ffill')

        #finding the wind speed in x and y direction
        #self.data = self.data.replace(360.0, 0) #There is some issue converting 360 deg to radians, therefore replacing 360 with 0
    
        x_wind = []
        y_wind = []

        earth_radius = 6271.0
        degrees_to_radians = math.pi/180.0
        radians_to_degrees = 180.0/math.pi
    
        for i in range(np.size(self.data['WSPD(knot)'])):
        
            x_wind.append(-math.cos(self.data["DRCT(deg)"][i]*degrees_to_radians)*self.data["WSPD(knot)"][i])
            y_wind.append(-math.sin(self.data["DRCT(deg)"][i]*degrees_to_radians)*self.data["WSPD(knot)"][i])
   
        self.data['x_wind'] = x_wind   
        self.data['y_wind'] = y_wind 
   
        new_lat = []
        new_long = []
    
        #Because of the nan values
        new_lat.append(self.lat)
        new_long.append(self.lon)

        old_x = 0
        old_y = 0

       
        #finding the new x and y positions
        lat =self.lat
        long=self.lon
        for k in range(1,np.size(self.data['x_wind'])):
        
            new_x = old_x + ((self.data['height(m)'][k]/5)-(self.data['height(m)'][k-1]/5))*(self.data['x_wind'][k] + self.data['x_wind'][k-1])/2
            new_y = old_y + ((self.data['height(m)'][k]/5)-(self.data['height(m)'][k-1]/5))*(self.data['y_wind'][k] + self.data['y_wind'][k-1])/2
        
            dx = new_x-old_x
            dy = new_y-old_y
  
            #change in latitude is the change in x along the north south line
            lat = lat - dx/(earth_radius*1000)*radians_to_degrees
            new_lat.append(lat)
        
            #change in longitude is the change in y along the east west line
        
            r = earth_radius*math.cos(lat*degrees_to_radians)
            long = long - (dy/(r*1000))*radians_to_degrees
            new_long.append(long)

            coords_2 = (new_lat[0], new_long[0])
            coords_1 = (lat, long)
   
            distance = geopy.distance.geodesic(coords_2, coords_1).m  
            self.distance.append(distance)            
            old_x = new_x
            old_y = new_y
    
        self.data['new_lat'] = new_lat
        self.data['new_long'] = new_long
        coords_2 = (new_lat[0], new_long[0])

        max_height=np.where(self.data["height(m)"]<7000)
        limit = max_height[0].max()

        coords_1 = (new_lat[limit], new_long[limit])
        self.h_disp = geopy.distance.geodesic(coords_2, coords_1).m

        #removing datapoints below 10 meters for 
        if self.data["height(m)"][0]<10:

            self.data=self.data.drop(self.data.index[0])

    

    def find_data_no_horizontal_disp(self):

    
        #converting knots into meters per second
        self.data["DRCT(deg)"] = self.data["DRCT(deg)"].fillna(method='bfill')
        self.data["DRCT(deg)"] = self.data["DRCT(deg)"].fillna(method='ffill')
    
        self.data["height(m)"] = self.data["height(m)"].fillna(method='bfill')
        self.data["height(m)"] = self.data["height(m)"].fillna(method='ffill')

        self.data["WSPD(knot)"] = self.data["WSPD(knot)"].fillna(method='bfill')
        self.data["WSPD(knot)"] = self.data["WSPD(knot)"].fillna(method='ffill')

        #finding the wind speed in x and y direction
    
        x_wind = []
        y_wind = []

        earth_radius = 6271.0
        degrees_to_radians = math.pi/180.0
        radians_to_degrees = 180.0/math.pi
    
        for i in range(np.size(self.data['WSPD(knot)'])):
        
            x_wind.append(-math.cos(self.data["DRCT(deg)"][i]*degrees_to_radians)*self.data["WSPD(knot)"][i])
            y_wind.append(-math.sin(self.data["DRCT(deg)"][i]*degrees_to_radians)*self.data["WSPD(knot)"][i])
   
        self.data['x_wind'] = x_wind   
        self.data['y_wind'] = y_wind 
   
        new_lat = []
        new_long = []
    
        #Because of the nan values
        new_lat=[self.lat]*len(x_wind)
        new_long=[self.lon]*len(x_wind)


    
        self.data['new_lat'] = new_lat
        self.data['new_long'] = new_long
       
       #Removing datapoints below 10 meter for the interpolations
        if self.data["height(m)"][0]<10:

            self.data=self.data.drop(self.data.index[0])


if __name__=="__main__":


    filename = "/home/norah/master/RS_data/RS_00_01_05_1986/BY_26702_54.7_20.62.txt"
    b = datetime64('1986-05-01T00:00:00.000000000')
    Radiosonde1 = Radiosonde(filename, 54.7,20.62,b)
    Radiosonde1.find_horizontal_disp()  
    print(Radiosonde1.data)
   



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


class Radiosonde_csv:

    """
    Innput
    ------------------------------------------------------------------------
    filename is the name of the csv file containing information of a radiosonde. 
    lat and long is the initial longitiude and latitude of the radiosonde.
    ------------------------------------------------------------------------

    Output
    ------------------------------------------------------------------------
    A cleaned dataframe containing height
    ------------------------------------------------------------------------
    """
    
    # Instance attribute
    def __init__(self, filename, time):
        
        """
        Innput
        ------------------------------------------------------------------------
        filename is the name of the dataset containing information of a radiosonde
        (has to be a .csv file). 

        time: datetime64
        ------------------------------------------------------------------------

        """

        self.data = pd.read_csv(filename)

   
        self.data["height"] = self.data["height"].interpolate()
      
        self.data = self.data[self.data['height'].notna()]
        self.data = self.data[self.data['windDirection'].notna()]
        self.data = self.data[self.data['windSpeed'].notna()]

        self.data=self.data.reset_index(drop=True)
     
        self.lat= self.data["latitude"][0]
        self.lon = self.data["longitude"][0]
        self.time = [time+np.timedelta64(int(height/5), 's') for height in self.data["height"]]
        self.data["time"] = self.time
        self.distance: list[float] = [0]
        self.h_disp = 1


    def find_horizontal_disp(self):

    
        x_wind = []
        y_wind = []

        earth_radius = 6271.0
        degrees_to_radians = math.pi/180.0
        radians_to_degrees = 180.0/math.pi
    
        for i in range(np.size(self.data['windSpeed'])):
        
            x_wind.append(-math.cos(self.data["windDirection"][i]*degrees_to_radians)*self.data["windSpeed"][i])
            y_wind.append(-math.sin(self.data["windDirection"][i]*degrees_to_radians)*self.data["windSpeed"][i])
   
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
        
            new_x = old_x + ((self.data['height'][k]/5)-(self.data['height'][k-1]/5))*(self.data['x_wind'][k] + self.data['x_wind'][k-1])/2
            new_y = old_y + ((self.data['height'][k]/5)-(self.data['height'][k-1]/5))*(self.data['y_wind'][k] + self.data['y_wind'][k-1])/2
        
            dx = new_x-old_x
            dy = new_y-old_y
            #self.distance.append(math.sqrt(dx**2+dy**2))
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
    
        self.data['latitude'] = new_lat
        self.data['longitude'] = new_long
        coords_2 = (new_lat[0], new_long[0])

        max_height=np.where(self.data["height"]<7000)
        limit = max_height[0].max()

        coords_1 = (new_lat[limit], new_long[limit])
        self.h_disp = geopy.distance.geodesic(coords_2, coords_1).m

        #removing datapoints below 10 meters for 
        if self.data["height"][0]<10:

            self.data=self.data.drop(self.data.index[0])


    def make_exner_levs(self, surface_p):

       
        self.data["exner"] =  np.power(self.data["pressure"]/surface_p, 0.286)



if __name__=="__main__":


    filename = "/home/norah/master/Rs_data_bufr/1986050512/51.4_6.97.csv"
    b = datetime64('1986-05-05T12:00:00.000000000')
    Radiosonde1 = Radiosonde_csv(filename, b)
   
    #Radiosonde1.find_horizontal_disp()  
    print(Radiosonde1.data)
    
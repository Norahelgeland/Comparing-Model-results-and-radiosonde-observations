"""
author: Nora Helgeland
date: May, 2023

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import matplotlib.cbook as cbook
from numpy import nan



def main():

    df=pd.read_excel('sonderfindings.xls', index_col=False )
  
    df1 = df[(df[3].str.contains("N")) & (df[4].str.contains("E"))]
    df2 = df[(df[3].str.contains("N")) & (df[4].str.contains("W"))]

    df1[3] = df1[3].str.replace("N","")
    df1[4] = df1[4].str.replace("E","")

  
    df2[3] = df2[3].str.replace("N","")
    df2[4] = df2[4].str.replace("W","")

    df1[3] = df1[3].astype(float)
    df1[4] = df1[4].astype(float)

    df2[3] = df2[3].astype(float)
    df2[4] = df2[4].astype(float)

    #df2[3] = -df2[3]
    df2[4] = -df2[4]
    
    df = pd.concat([df1,df2])
 
    lat_min=45
    lat_max=70
    long_min=-16
    long_max=38

    lat_mask = df[3].between(lat_min, lat_max)
    long_mask = df[4].between(long_min, long_max)

    df = df[lat_mask & long_mask]
    
    df.to_csv('filtered_stations.csv', index=False)



if __name__=="__main__":

    main()





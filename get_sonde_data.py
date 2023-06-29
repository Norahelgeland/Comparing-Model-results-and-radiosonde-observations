"""
author: Brian Blaylock
date: July 6, 2015

script retrieved from: https://kbkb-wx-python.blogspot.com/2015/07/plotting-sounding-data-from-university.html
"""

import urllib3 as urllib2
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
#from skewt import SkewT


def GetRSdata(name,stn, year, month, day, hour, lat, lon):


    # 1)
    # Wyoming URL to download Sounding from
    url = 'http://weather.uwyo.edu/cgi-bin/sounding?region=naconf&TYPE=TEXT%3ALIST&YEAR='+year+'&MONTH='+month+'&FROM='+day+hour+'&TO='+day+hour+'&STNM='+stn
    r = requests.get(url)
    content = r.text
    print(content)

    # 2)
    # Remove the html tags
    soup = BeautifulSoup(content)
    data_text = soup.get_text()

    # 3)
    # Split the content by new line.
    splitted = list(data_text.split('\n'))
    
    encountered = None
    endline = None
    for skip, line in enumerate(splitted):
        if "-----------------------------------------------------------------------------" in line:
            encountered = skip
            break
    
    for stop, line in enumerate(splitted):
        if "Station information and sounding indices" in line:
            endline = stop
            break

    
    if encountered is None:
        return
    
    for stop, line in enumerate(splitted):
        if "Station information and sounding indices" in line:
            endline = stop
            break
  
    header = splitted[encountered+1]
    data_text = splitted[encountered+4:endline]
    # 4)
    # Write this splitted text to a .txt document
    Sounding_filename = "RS_data/RS"+"_"+str(hour)+"_"+str(day)+"_"+str(month)+"_"+str(year)+"/"+str(name)+"_"+str(stn)+"_"+str(lat)+'_'+str(lon)+'.txt'
    f = open(Sounding_filename,'w')
    f.write(header+'\n')
    for line in data_text:
        f.write(line+'\n')
    f.close()



def main():

    df = pd.read_csv('filtered_stations.csv')
    
    print(df)

    year= '1986'
    month = '04'
    day = '26'
    hour = '00' #either 12 or 00

    N = np.size(df["2"])

    for i in range(1,N):

        stn = str(df["2"][i])
        name = str(df["9"][i])
        lat = str(df["3"][i])
        lon = str(df["4"][i])
        GetRSdata(name, stn, year, month, day, hour, lat, lon)


if __name__ == "__main__":

    main()

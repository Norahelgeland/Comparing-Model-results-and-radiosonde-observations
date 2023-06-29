# Comparing Model Results and Radiosonde Observations

## extract bufr file and place in separate csv files

bufr_extract.ipynb is used to place the data from the bufr files(at each timestep) in csv files corresponding to each radiosonde sounding. The date has to be changed to retrieve the correct bufr file. Also the desired area has to be defined.

## Interpolation to compare model with radiosonde observations from bufr file

**Interpolate_exner.py** is used to interpolate the model data in exner coordinates and time to have the same position vertically and temporally as the atmospheric soundings retrieved from the bufr files. Only the dates in the main function needs to be changed between runs. The filenames also need to be changed. The corresponding folders need to be created in the correct location before running. The jobs are submitted to the supercomputer through the **job_submit.sh** file. To run it on your own computer some modifications need to be made in the paths. **radiosonde_class2.py** is used to make radiosonde objects by retriving the already produced csv files and producing a cleaned pandas dataset and need to be i n the correct folder to be imported. A class function is called for in the main to create xner levels corresponding to the specific atmospheric sounding

##  retrieve atmospheric sounding data from the wyoming web page

**get_sonde_data.py** is used to retrieve data from the wyoming university web page. **filteres_stations.py** are used to tell the script what to look for. One text file per atmospheric sounding is then placed i a folder corresponding to the time and date. To retrieve data from another data, the corresponding folder need to be created and the time and data in the main function in **get_sonde_data.py** need to be changed.

## Interpolation in 4d to compare model with atmospheric soundings retrieved from the Wyoming University webpage

Interpolation_4d_height.py is used to interpolate the model in height, xy and time to have the same position in space and time. Since the model data does not contain height, geopotential height in each hybrid level is calculated in **make_geopot_levs.py**. By using the **job_submit.sh** and pasting the path to the **make_geopot_levs.py**  file the geopotential height will be calculated for the whole model grid in each height for six hours in each timestep. This has to be done before running the **Interpolation_4d_heightTh**is file is run in the same way as explained for the interpolation_exner.py file. The radiosnde class for this interpolation is in **radiosonde_class.py** and need to be imported correctly.

## Statistical analysis

In **Statistical_analysis.py**:

- polar plots showing wind direction
- violinplots showing mean abolute errror and mean error 
- Scatterplots

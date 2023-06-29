# Comparing Model Results and Radiosonde-Observations

## extract bufr file and place in separate csv files

bufr_extract.ipynb is used to place the data from the bufr files(at each timestep) in csv files corresponding to each radiosonde sounding. The date has to be changed to retrieve the correct bufr file. Also the desired area has to be defined.

## Interpolation to compare model with radiosonde observations from bufr file

**Interpolate_exner.py** is used to interpolate the model data in exner coordinates and time to have the same position vertically and temporally as the atmospheric soundings retrieved from the bufr files. Only the dates in the main function needs to be changed between runs. The filenames also need to be changed. The corresponding folders need to be created in the correct location before running. The jobs are submitted to the supercomputer through the **job_submit.sh** file. To run it on your own computer some modifications need to be made in the paths. **radiosonde_class2.py** is used to make radiosonde objects by retriving the already produced csv files and producing a cleaned pandas dataset and need to be i n the correct folder to be imported. A class function is called for in the main to create xner levels corresponding to the specific atmospheric sounding

## Interpolation in 4d to compare model with atmospheric soundings retrieved from the Wyoming University webpage

Interpolation_4d_height.py is used to interpolate the model in height, xy and time to have the same position in space and time 

## Statistical analysis

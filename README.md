# Comparing Model Results-and-Radiosonde-Observations

## extract bufr file and place in separate csv files

bufr_extract.ipynb is used to place the data from the bufr files(at each timestep) in csv files corresponding to each radiosonde sounding. The date has to be changed to retrieve the correct bufr file. Also the desired area has to be defined.

## Interpolation to compare model with bufr file

Interpolate_exner.py is used to interpolate the model data to have the same position vertically and temporally. Only the dates in the main function needs to be changed between runs. The filenames also need to be changed. The corresponding folders need to be created in the correct location before running. The jobs are submitted to the supercomputer through the job.sh file. To run it on your own computer some modifications need to be made in the paths.

## Interpolation in 4d to compare model with atmospheric soundings retrieved from the Wyoming University webpage

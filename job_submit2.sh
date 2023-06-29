#! /bin/bash
#$ -l h_rt=72:00:00
#$ -l h_rss=2G,mem_free=2G,s_rss=1.9G
#$ -q research-r8.q
#$ -M norah@met.no
#$ -m a
##$ -pe smp 1
#$ -R y
#$ -S /bin/bash
#$ -v PWD
#$ -cwd
#$ -t 1:26  # Number of jobs
#$ -N geopot_levs

echo hello to id $SGE_TASK_ID

source /modules/rhel8/conda/install/bin/activate
conda activate master3

#rm geopot_levs_*


/home/norah/.conda/envs/master3/bin/python /home/norah/master/python_files/make_geopot_levs.py

#/home/norah/.conda/envs/master3/bin/python /home/norah/master/python_files/interpolation.py
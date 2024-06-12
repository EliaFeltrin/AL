#!/bin/bash

# Bash to profile the exe and generate the roofline model   

#check that there is at least two argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <kernel_version_name>"
    exit 1
fi

# cd.. then run profile_nvp.sh
cd ..
bash compile.sh compile
sudo bash ./GPU-Roofline-Python/compiler_metrics/profile_nvp.sh

#move the events_exe.csv, metrics_exe.csv, timing_exe.csv to the compiler_metrics folder then give them permission to be read and written
sudo mv events_exe.csv GPU-Roofline-Python/compiler_metrics/
sudo mv metrics_exe.csv GPU-Roofline-Python/compiler_metrics/
sudo mv timing_exe.csv GPU-Roofline-Python/compiler_metrics/

sudo chmod a+rw ./GPU-Roofline-Python/compiler_metrics/events_exe.csv
sudo chmod a+rw ./GPU-Roofline-Python/compiler_metrics/metrics_exe.csv
sudo chmod a+rw ./GPU-Roofline-Python/compiler_metrics/timing_exe.csv

#run the python script to generate the roofline model
source virtualenv/bin/activate
cd GPU-Roofline-Python
python roofline_tool.py $1

# delete the events_exe.csv, metrics_exe.csv, timing_exe.csv
# sudo rm ./compiler_metrics/events_exe.csv
# sudo rm ./compiler_metrics/metrics_exe.csv
# sudo rm ./compiler_metrics/timing_exe.csv

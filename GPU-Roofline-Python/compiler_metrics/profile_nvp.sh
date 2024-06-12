#!/bin/bash

# Script profiling a CUDA application (transpose) 
# A python application can be profiled likewise
# the nvprof profiler is used

exe="exe" # modify appropriately
flags="-mN 25 -MN 25 -mM 4 -MM 4 -ml=0 -mu 0.1 -l -a 1000 -i 1 -r 0.1 -PCRq 0.3,2,1,2 -c -s -f"

# metrics for nvprof
metrics='inst_executed_global_loads,gld_transactions,gst_transactions,shared_load_transactions,shared_store_transactions,l2_read_transactions,l2_write_transactions,dram_read_transactions,dram_write_transactions'

events='thread_inst_executed'
# if profiling python module it may need the following option
# options='--openacc-profiling off'

nvprof --csv --print-gpu-summary --log-file timing_${exe}.csv  ./${exe} ${flags}
nvprof  --csv --metrics $metrics --log-file metrics_${exe}.csv ./${exe} ${flags}
nvprof  --csv --events $events   --log-file events_${exe}.csv ./${exe} ${flags}

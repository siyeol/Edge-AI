#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J HW1_gr0                        # Job name
#SBATCH -o HW1_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW1_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                           # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-user=endri.taka@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A EE379K                 # Allocation name

# Other commands must follow all #SBATCH directives...

module load intel/18.0.2 python3/3.7.0
module load cuda/10.0 cudnn/7.6.2 nccl/2.4.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.0/lib64
source $WORK/HW1_virtualenv/bin/activate

# Launch code...
#python $WORK/HW1_files/starter.py --batch_size=128 --epochs=25 --lr=0.01 > $WORK/HW1_files/out


#python3 p3_q2.py --epochs=10 --conv1_ch_size=8 --conv2_ch_size=16 > $WORK/HW1_files/out_p3_q2_ep_10_conv1_8_conv2_16

python3 p3_q2.py --epochs=15 --conv1_ch_size=8 --conv2_ch_size=16 > $WORK/HW1_files/out_p3_q2_ep_15_conv1_8_conv2_16

python3 p3_q2.py --epochs=20 --conv1_ch_size=8 --conv2_ch_size=16 > $WORK/HW1_files/out_p3_q2_ep_20_conv1_8_conv2_16

python3 p3_q2.py --epochs=25 --conv1_ch_size=8 --conv2_ch_size=16 > $WORK/HW1_files/out_p3_q2_ep_25_conv1_8_conv2_16


#python3 p3_q2.py --epochs=10 --conv1_ch_size=8 --conv2_ch_size=32 > $WORK/HW1_files/out_p3_q2_ep_10_conv1_8_conv2_32

# ---------------------------------------------------


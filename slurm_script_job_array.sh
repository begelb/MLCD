#!/bin/bash

#SBATCH --job-name=run_job_array   # Job name
#SBATCH --output=tmp_output_radial3/output_%N_%j.out  # STDOUT output file
#SBATCH --error=tmp_output_radial3/output_%N_%j.err   # STDERR output file (optional)
#SBATCH --partition=p_mischaik_1  # Partition (job queue)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1          # Number of CPUs (cores) per task (>1 if multithread tasks)
#SBATCH --mem=1000                  # Real memory (RAM) required (MB)
#SBATCH --array=0-399          # Array job will submit 100 jobs
#SBATCH --time=20:00:00            # Total run time limit (hh:mm:ss)
#SBATCH --requeue                  # Return job to the queue if preempted
#SBATCH --export=ALL               # Export you current env to the job env

# Load necessary modules
# module purge
# module load intel/19.0.3
cd /home/bg545/attractor_id/attractor_identification_draft/

# Run python script with input data. The variable ${SLURM_ARRAY_TASK_ID}
# is the array task id and varies from 0 to 99 in this example.
srun python3 /home/bg545/attractor_id/attractor_identification_draft/run_experiment.py ${SLURM_ARRAY_TASK_ID}
# /home/bg545/Research/Final_Reg_Training_Homology_colormap.py
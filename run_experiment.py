from .src.attractor_id.config import configure
from .src.attractor_id.experiment import Experiment
import sys 

''' This is a script to run an experiment using Slurm Workload Manager, specifically the file slurm_script_job_array.sh '''

''' Global variables set by user '''

''' 
The following systems are implemented:
- 'curved_separatrix'
- 'ellipsoidal_2d'
- 'ellipsoidal_3d'
- 'ellipsoidal_larger_domain_4d'
- 'ellipsoidal_larger_domain_5d'
- 'EMT'
- 'periodic'
- 'radial_2labels'
- 'radial_3labels'
- 'straight_separatrix'
'''

system = 'straight_separatrix'

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.49]

# repetitions_per_parameter_set is the number of nodes being used in the cluster
# so, if line 11 of slurm_script_job_array.sh is #SBATCH --array=0-499, then repetitions_per_parameter_set should be 500
repetitions_per_parameter_set = 100

''' Main code block '''

def main():
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    N_list = config.N_list
    experiment_class = Experiment(N_list)
    job_index = int(sys.argv[1]) # job index is read from the job_array controlled by slurm_script_job_array.sh
    experiment_class.run_experiment(job_index, system, repetitions_per_parameter_set, labeling_threshold_list)
if __name__ == "__main__":
    main()


    
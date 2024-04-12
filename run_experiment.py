from src.attractor_id.config import configure
from src.attractor_id.experiment import Experiment
import sys 

''' Global variables set by user '''

# system is an integer that refers to which dynamical system the user would like to use
system = 'periodic'

''' 
The following systems are implemented:
- 'straight_separatrix'
- 'radial_2labels'
- 'radial_3labels'
- 'curved_separatrix'
- 'EMT'
- 'periodic'
- 'ellipsoidal_2d'
- 'DSGRN_2d_network' 
'''


# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.1, 0.2, 0.3, 0.4, 0.49]

# repetitions_per_parameter_set is the number of nodes being used in the cluster
# so, if line 13 of slurm_script_job_array.sh is #SBATCH --array=0-499, then repetitions_per_parameter_set should be 500
repetitions_per_parameter_set = 500

''' Global variables that should not be changed by the user '''
# job index is read from the job_array controlled by slurm_script_job_array.sh
job_index = int(sys.argv[1])

''' Main code block '''

def main():
    config_fname = f'config/{system}.txt'
    config = configure(config_fname)
    N_list = config.N_list
    experiment_class = Experiment(N_list)
    experiment_class.run_experiment(job_index, config, repetitions_per_parameter_set, labeling_threshold_list)

if __name__ == "__main__":
    main()
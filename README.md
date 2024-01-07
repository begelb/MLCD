# attractor-identification-draft

## Requirements
This repository has several required Python packages. The requirements are listed in requirements.txt. To install all of the requirements at once, run the following command:

```pip install -r requirements.txt```

For homology computations, we use the Python extension [pyCHomP2](https://pypi.org/project/pychomp2/).


## How to run the code for a single example
To compute a single example, run single_example.py. At the top of the file, under "Global variables set by user", you can change:
- the system
- the number of nodes in the hidden layer of the neural network
- the list of labeling thresholds, and
- the integer name that refers to the example.

Other variables, which we expect to be changed less frequently--such as learning rate and optimizer choice for the neural network--are specified in the txt files located in the folder config.

## How to run an experiment
To run an experiment on the Amarel cluster, use the shell script amarel_cluster_code/slurm_script_job_array.sh and run_experiment.py. At the top of the file, under "Global variables set by user", you can change the system, the list of labeling thresholds, and repetitions per unique set of parameters. The lists of hidden layer widths for each system are specified in the txt files located in the folder config.
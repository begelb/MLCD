# attractor-identification-draft

## Requirements
This software was developed using [Python 3.9.6](https://www.python.org/downloads/release/python-396/).

The software has several required Python packages. The requirements are listed in requirements.txt. To install all of the requirements at once, run the following terminal command:

```pip install -r requirements.txt```

For homology computations, we use the Python extension [pyCHomP2](https://pypi.org/project/pychomp2/).

## How to run the code for a single example
To compute a single example, run ``` single_example.py ```. At the top of the file, under "Global variables set by user", you can change:
- the system,
- the number of nodes in the hidden layer of the neural network,
- the list of labeling thresholds, and
- the integer name that refers to the example.

Other variables, which we expect to be changed less frequently--such as learning rate and optimizer choice for the neural network--are specified in the ``` .txt ``` files located in the folder ```config``` and numbered by the corresponding system.

## How to run an experiment
To run an experiment on the Amarel cluster, use the shell script ``` amarel_cluster_code/slurm_script_job_array.sh ```. The path in line 25 should be appropriately modified to use your username and the folder where the code of this repository is located. At the top of ``` run_experiment.py ```, under "Global variables set by user", you can change the system, the list of labeling thresholds, and repetitions per unique set of parameters. The lists of hidden layer widths for each system are specified in the txt files located in the folder ```config```.

## Acknowledgements
The file ```amarel_cluster_code/slurm_script_job_array.sh``` is taken from the GitHub repository [cluster-help](https://github.com/marciogameiro/cluster-help) written by Marcio Gameiro (2020) and available under MIT License. 

The configuration set-up is based on the GitHub repository [MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space](https://github.com/Ewerton-Vieira/MORALS/tree/main) written by Ewerton Vieira, Aravind Sivaramakrishnan, and Sumanth Tangirala (2023) and available under MIT License.

'''


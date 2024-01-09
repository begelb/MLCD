# attractor-identification-draft

## Requirements
This code was developed using [Python 3.9.6](https://www.python.org/downloads/release/python-396/).

Compiling the code requires several Python packages. The requirements are listed in ```requirements.txt.``` To install all of the requirements at once after cloning the respository, use the following command:

```pip install -r requirements.txt```

For homology computations, we use the Python extension [pyCHomP2](https://pypi.org/project/pychomp2/).

## System numbers

See ```data/system_number_meanings.txt``` for a dictionary of the system integer labels to the qualitative descriptions of these systems. 

## How to create your own data
Data for each system is already produced and contained in the ```data``` directory.
To produce new data, one can run ```data_production/make_data.py```. Choose the system number and the number of initial points, and a persistence diagram will be produced. From this, you will need to choose an appropriate threshold, and then the data will be saved as '''data.csv''', which one can split into a training/testing set.

## How to run the code for a single example
To compute a single example, run ``` single_example.py ```. 

At the top of the file, under "Global variables set by user", you can change:
- the system number,
- the number of nodes in the hidden layer of the neural network,
- the list of labeling thresholds, and
- the integer name that refers to the example.

Other variables, which we expect to be changed less frequently--such as learning rate and optimizer choice for the neural network--are specified in the ``` .txt ``` files located in the folder ```config``` and numbered by the corresponding system. This computation will use the training and testing datasets with paths specified in the configuration file. By default, the paths are specified with the provided data.

### Figures
If the dimension is two and ```make_figures``` is set to True in the configuration file, then figures will be saved in ```output/figures```.

### Models
Models will be saved in ```output/models```.

### Homology results
The homology results will be saved in ```output/results```.


## How to run an experiment
To run the experiments described in the paper, we used [Slurm Workload Manager](https://slurm.schedmd.com/overview.html). To run your own experiments, you can modify the script ``` amarel_cluster_code/slurm_script_job_array.sh ``` and use the slurm command ```sbatch```. For more detail, see the [slurm documentation](https://slurm.schedmd.com/sbatch.html). At the top of ```run_experiment.py```, under "Global variables set by user", you can change:
- the system number,
- the list of labeling thresholds, and
- repetitions per unique set of parameters.

By default, the only parameter that is varied is the width of the neural network hidden layer. The lists of hidden layer widths for each system are specified in the ```.txt``` files located in the folder ```config```.

## Acknowledgements
- The file ```amarel_cluster_code/slurm_script_job_array.sh``` was copied from the GitHub repository [cluster-help](https://github.com/marciogameiro/cluster-help) written by **Marcio Gameiro** (2020) and available under MIT License. Small modifications to the file were made. See LICENSE.md for copyright information pertaining to this file. 

- The file ```src/config.py``` and the files inside ```config``` are based on the contents of the GitHub repository [MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space](https://github.com/Ewerton-Vieira/MORALS/tree/main) written by **Ewerton Vieira, Aravind Sivaramakrishnan, and Sumanth Tangirala** (2023) and available under MIT License. See LICENSE.md for copyright information pertaining to these files.

- The authors acknowledge the [Office of Advanced Research Computing (OARC)](https://oarc.rutgers.edu) at Rutgers, The State University of New Jersey for providing access to the Amarel cluster and associated research computing resources that have been used to develop and run this code.

- [GitHub Copilot](https://github.com/features/copilot), developed by **GitHub, OpenAI, and Microsoft** (2024), was used in the development of this code with Duplicate Detection filtering feature set to “Block." See the [GitHub Copilot FAQ](https://github.com/features/copilot) for further information about copyright and training data. 

- The files ```network.py``` and ```train.py``` contain modified source code from [PyTorch Tutorials](https://github.com/pytorch/tutorials) written by **Pytorch contributors** (2017-2022). See LICENSE.md for copyright information pertaining to these files.

- [Visual Studio Code](https://code.visualstudio.com), developed by **Microsoft** (2024), was used in the development of this code.

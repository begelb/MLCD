# Machine-Learned Cubical Decomposition (MLCD) of Phase Space for Identifying Attracting Neighborhoods

This code accompanies the paper "Data-driven Identification of Attractors using Machine Learning" by  Marcio Gameiro, Brittany Gelb, William D. Kalies, Miroslav Kramar, Konstantin Mischaikow, and Paul Tatasciore.

## Google Colab: run MLCD with no local installation
The code is available on a [Google colab notebook](https://colab.research.google.com/drive/1teYPxaoI0IuQCSlEQuJ9zy43vhqGUqWD?usp=sharing), which does not require local installation to run. The data is automatically read from [a Github repository that hosts the data](https://github.com/begelb/MLCD-data). 

## Installation
This code was developed using [Python 3.9.6](https://www.python.org/downloads/release/python-396/). **The program must be executed using Python<=3.11**.

After downloading a copy of the repository, navigate to the project folder and use one of the following commands to install the attractor_id package. The first commmand is preferred if you have multiple versions of Python. 

**Installation Command Option 1 (preferred)**: 
```python -m pip install .```

**Installation Command Option 2**:
```pip install .```


## Running the code locally in a Jupyter notebook
The main functionality of the code is available in ```learn_decomposition.ipynb```. In this Jupyter noteboook, it is possible to:
- Train a cubical neural network using the data from the paper
- Produce a plot of the learned function and cubical decomposition of phase space,
- Produce a plot of the labeled cubes, and
- Compute homology of labeled regions that are approximations of attracting neighborhoods.

## How to create your own data
Data for each system is already produced and contained in the ```data``` directory.
To produce new data, one can run ```data_production/make_data.py```. Choose the system number and the number of initial points, and a persistence diagram will be produced. From this, it is necessary to choose an appropriate threshold, and then the data will be saved as ```data.csv```, which one can split into a training and testing set.

## System configurations
Variables that are system specific or which we expect to be changed infrequently--such as learning rate and optimizer choice for the neural network--are specified in the ``` .txt ``` files located in the folder ```config``` and named by the corresponding system. With the exception of the Jupyter notebook, ```learn_decomposition.ipynb```, all computations use local copies of the training and testing datasets with paths specified in the configuration file. By default, the paths are specified with the provided data.

The variable data_bounds specifies the domain of the data. The final decomposition of phase space is intersected with this domain. The set $[0, 1] \times [0, 1]$ should be written as $[[0, 1], [0, 1]]$.

If ```threshold_prediction = True```, then as in the paper, the result of the neural network is thresholded between $0$ and $K-1$ where $K$ is the number of labels.

If ```weak_weight_share = True```, then the network is constrained so that parallel hyperplanes correspond to weights that vary by *trainable* constant multiples, which are initialized all equal to one. If ```weak_weight_share = False```, then these constant multiples are *not trainable* and thus remain equal to one throughout the duration of training. The results of the paper were produced with ```weak_weight_share = False```.

The parameter ```reduction_threshold``` corresponds to $\beta$ in the paper: it is the proportional decrease of the training loss between the first and final epoch that must be observed in order to continue the computation to the step of obtaining the decomposition. If this criterion is not met, the training restarts (at different initial conditions). For example, setting reduction_threshold = 0.1 means that the training loss must decrease by 10% throughout the duration of training in order to continue.

The parameter ```patience``` corresponds to $\rho$ in the paper: it controls an early stopping criterion for the training process.

## Alternative way to compute a single example
As an alternative to the Jupyter notebook, it is possible to compute a single example using the file ``` single_example.py ```. 

At the top of the file, under "Global variables set by user", it is possible to change:
- the system name,
- the number of nodes in the hidden layer of the neural network,
- the list of labeling thresholds, and
- the integer name that refers to the example.

### Figures
If the dimension is two and ```make_figures``` is set to True in the configuration file, then figures will be saved in ```output/figures```.

### Models
Models will be saved in ```output/models```.

### Homology results
The homology results will be saved in ```output/results```.

## Acknowledgements
- The file ```slurm_script_job_array.sh``` was copied from the GitHub repository [cluster-help](https://github.com/marciogameiro/cluster-help) written by **Marcio Gameiro** (2020) and available under MIT License. Small modifications to the file were made. See LICENSE.md for copyright information pertaining to this file.

- The functions used to produce the stacked histograms are copied from [the matplotlib documentation](https://matplotlib.org/stable/gallery/lines_bars_and_markers/filled_step.html)

- The file ```src/config.py``` and the files inside ```config``` are based on the contents of the GitHub repository [MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space](https://github.com/Ewerton-Vieira/MORALS/tree/main) written by **Ewerton Vieira, Aravind Sivaramakrishnan, and Sumanth Tangirala** (2023) and available under MIT License. See LICENSE.md for copyright information pertaining to these files.

- The authors acknowledge the [Office of Advanced Research Computing (OARC)](https://oarc.rutgers.edu) at Rutgers, The State University of New Jersey for providing access to the Amarel cluster and associated research computing resources that have been used to develop and run this code.

- [GitHub Copilot](https://github.com/features/copilot), developed by **GitHub, OpenAI, and Microsoft** (2024), was used in the development of this code with Duplicate Detection filtering feature set to “Block." See the [GitHub Copilot FAQ](https://github.com/features/copilot) for further information about copyright and training data. 

- The files ```network.py``` and ```train.py``` contain modified source code from [PyTorch Tutorials](https://github.com/pytorch/tutorials) written by **Pytorch contributors** (2017-2022). See LICENSE.md for copyright information pertaining to these files.

- [Visual Studio Code](https://code.visualstudio.com), developed by **Microsoft** (2024), was used in the development of this code.

## Explanation of requirements
- To run ```data_production/make_data.py```, the user must have ```matplotlib<=3.5.6``` due to a [known issue with gudhi and matplotlib](https://github.com/GUDHI/gudhi-devel/issues/724).

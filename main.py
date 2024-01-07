from src.compute_example import compute_example
from src.config import configure

'''' Global variables set by user '''

# system is an integer that refers to which dynamical system the user would like to use
system = 1

# N is the number of nodes in the hidden layer of the network
N = 4

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.1, 0.3]

# example index gives an integer-valued name to the computation, which corresponds to file names in the folder output
example_index = 0

''' Main code block '''

def main():
    config_fname = f'config/system{system}.txt'
    config = configure(config_fname)
    compute_example(config, example_index, N, labeling_threshold_list)

if __name__ == "__main__":
    main()

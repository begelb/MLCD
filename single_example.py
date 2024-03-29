from src.attractor_id.compute_example import compute_example

'''' Global variables set by user '''

# system is an integer that refers to which dynamical system the user would like to use
# system_name = 'straight_separatrix'
system_number = 1

# N is the number of nodes in the hidden layer of the network and must be an integer multiple of the dimension of the system
N = 10

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.1]

# example index gives an integer-valued name to the computation, which corresponds to file names in the folder output
example_index = 2

''' Main code block '''

def main():
    compute_example(system_number, N, labeling_threshold_list, example_index)

if __name__ == "__main__":
    main()

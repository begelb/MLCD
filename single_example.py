from src.MLCD.compute_example import compute_example

'''' Global variables set by user '''

''' 
Available systems:
- 'linear_separatrix' (2d)
- 'nonlinear_separatrix' (4d)
- 'radial_bistable' (2d)
- 'radial_tristable' (2d)
- 'M3D' (3d)
- 'hill_system_with_PO' (3d)
- 'ellipsoidal_bistable_2d'
- 'ellipsoidal_bistable_3d'
- 'ellipsoidal_bistable_4d'
- 'ellipsoidal_bistable_5d'
'''

system = 'linear_separatrix'

# N is the number of nodes in the hidden layer of the network and must be an integer multiple of the dimension of the system
N = 4

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.49]

# example index gives an integer-valued name to the computation, which corresponds to file names in the folder output
example_index = 4

''' Main code block '''

def main():
    compute_example(system, N, labeling_threshold_list, train_only = False, example_index = example_index)

if __name__ == "__main__":
    main()

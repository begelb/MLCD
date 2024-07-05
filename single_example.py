from src.attractor_id.compute_example import compute_example

'''' Global variables set by user '''

''' 
The following systems are implemented:
- 'straight_separatrix'
- 'radial_2labels'
- 'radial_3labels'
- 'curved_separatrix'
- 'EMT'
- 'periodic'
- 'ellipsoidal_2d'
- 'ellipsoidal_3d'
- 'ellipsoidal_4d'
- 'ellipsoidal_5d'
- 'ellipsoidal_6d'
- 'DSGRN_2d_network' 
- 'leslie'
- 'periodic_3labels'
- 'iris'
'''

system = 'straight_separatrix'

# N is the number of nodes in the hidden layer of the network and must be an integer multiple of the dimension of the system
N = 2

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.49]

# example index gives an integer-valued name to the computation, which corresponds to file names in the folder output
example_index = 4

''' Main code block '''

def main():
    compute_example(system, N, labeling_threshold_list, example_index)

if __name__ == "__main__":
    main()

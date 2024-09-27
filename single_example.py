from src.attractor_id.compute_example import compute_example

'''' Global variables set by user '''

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

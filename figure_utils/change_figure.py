from src.attractor_id.change_figure_src import change_figure

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

system = 'radial_2labels'
#system = 'ellipsoidal_2d'

# N is the number of nodes in the hidden layer of the network and must be an integer multiple of the dimension of the system

if system == 'radial_3labels':
    N = 8
    example_index = 7
    model_fname = 'paper_figures_with_models/radial_3labels/7-model.pth'
if system == 'radial_2labels':
    N = 4
    example_index = 4
    model_fname = 'paper_figures_with_models/radial_2labels/0-model.pth'

if system == 'ellipsoidal_2d':
    N = 4
    example_index = 2
    model_fname = 'paper_figures_with_models/ellipsoidal_2d/2-model.pth'

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.49]

# example index gives an integer-valued name to the computation, which corresponds to file names in the folder output
#example_index = 7

#model_fname = f'paper_figures_with_models/{system}/{example_index}-model.pth'

''' Main code block '''

def main():
    #change_figure(system, N, labeling_threshold_list, example_index, True, False, model_fname)
    change_figure(system, N, labeling_threshold_list, example_index, True, False, model_fname)
    #change_figure(system, N, labeling_threshold_list, example_index, True, True, model_fname)

if __name__ == "__main__":
    main()
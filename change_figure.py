from src.attractor_id.change_figure_src import change_figure

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

#system = 'ellipsoidal_2d'
#system = 'radial_2labels'
system = 'radial_3labels'
#system = 'straight_separatrix'

# N is the number of nodes in the hidden layer of the network and must be an integer multiple of the dimension of the system

if system == 'radial_3labels':
   # N = 8 for example_index = 7, N = 10 for example_index = 97
    N = 10
    example_index = 97
    model_fname = f'paper_figures_with_models/radial_3labels/{example_index}-model.pth'

if system == 'radial_2labels':
    N = 4
    example_index = 4
    model_fname = f'paper_figures_with_models/radial_2labels/0-model.pth'

if system == 'ellipsoidal_2d':
    N = 4
    example_index = 2
    model_fname = f'paper_figures_with_models/ellipsoidal_2d/{example_index}-model.pth'

if system == 'straight_separatrix':
    N = 2
    example_index = 4
    model_fname = f'paper_figures_with_models/straight_separatrix/0-model.pth'

# labeling threshold is the list of labeling thresholds to be used to label the cubes
labeling_threshold_list = [0.49]

''' Main code block '''

''' 
- decomposition
- polytopes
- data
- multicolor polytopes
'''

def main():
    change_figure(system, N, labeling_threshold_list, example_index, decomposition=True, polytopes=False, data=False, multicolor=False, model_file_name=model_fname)
    #change_figure(system, N, labeling_threshold_list, example_index, False, False, True, model_fname)
    #change_figure(system, N, labeling_threshold_list, example_index, False, True, False, model_fname)
    #change_figure(system, N, labeling_threshold_list, example_index, True, True, False, model_fname)

if __name__ == "__main__":
    main()

import csv
from src.attractor_id.histograms import plot_loss_histogram
import ast
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from matplotlib.ticker import FuncFormatter

def format_ticks(value, _):
    return f"{value:.3f}"  # Format tick labels to 2 decimal places

def plot_results(sorted_numbers, sorted_bools, file_path):
    """
    Plots a scatter plot where:
    - `sorted_numbers` is the sorted list of values.
    - `sorted_bools` determines the color (True = success, False = failure).
    """

    font = {'family' : 'serif',
            'size'   : 25}
    
    plt.figure(figsize=(8, 6))
    # straight_separatrix use 8.5

    plt.rc('font', **font)

    success_x = [i for i, label in enumerate(sorted_bools) if label]
    success_y = [num for num, label in zip(sorted_numbers, sorted_bools) if label]

    failure_x = [i for i, label in enumerate(sorted_bools) if not label]
    failure_y = [num for num, label in zip(sorted_numbers, sorted_bools) if not label]

        # Plot failures second (on top)
    plt.scatter(failure_x, failure_y, c='#440154FF', marker='^', s=60)

    # Plot successes first (background)
    plt.scatter(success_x, success_y, c='#7AD151FF', marker='+', s=80)

    # for i, (num, label) in enumerate(zip(sorted_numbers, sorted_bools)):
    #     color = '#7AD151FF' if label else '#440154FF'
    #     marker = 'o' if label else '^'  
    #     plt.scatter(i, num, c=color, marker=marker, s = 60)

# edgecolors = black
    plt.xlabel("Trial")
    plt.ylabel("Test loss")

   # plt.xticks([])  # Remove x-axis ticks
    plt.yticks(np.linspace(min(sorted_numbers), max(sorted_numbers), num=5))
    #plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    plt.xticks([0, 50, 100])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_ticks))

    # Add legend using dummy points
    plt.scatter([], [], c='#7AD151FF', marker='+', label="Success", s=150)
    plt.scatter([], [], c='#440154FF', marker='^', label="Failure", s=80)
    plt.legend()
    #plt.title("Scatter Plot of Successes and Failures")
   # plt.show()
    plt.tight_layout()
    plt.savefig(file_path)


# Example usage

system = 'M3D' #'ellipsoidal_5d_more_balanced_86_14' #'ellipsoidal_larger_domain_5d_N20'#'periodic_oversampled' #'periodic_oversampled' #'radial_3labels' #'radial_3labels'
#system = 'periodic_balanced' #'EMT_N6_1_28'#'ellipsoidal_larger_domain_5d_N20' #'periodic' #'ellipsoidal_larger_domain_4d_N16' #'radial_3labels' #'radial_3labels' #'ellipsoidal_2d' #'ellipsoidal_larger_domain_5d_N20' #'EMT_threshold_pred'

tmp_num = 91

if system == 'ellipsoidal_4d_more_balanced':
    num_jobs = 300
else:
    num_jobs = 100
    
if system == 'EMT' or system == 'EMT_threshold_pred' or system == 'EMT_corrected' or system == 'EMT_N6_1_28':
    correct_hom_uncertain = [1, 0, 0, 0, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0, 0, 0, 0]

if system == 'periodic' or system == 'periodic_balanced' or system == 'periodic_oversampled':
    correct_hom_uncertain = [1, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0]
    correct_hom_two = [1, 0, 0, 0]
    correct_hom_three = [1, 0, 0, 0]

if system == 'M3D':
    correct_hom_uncertain = [2, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0]
    correct_hom_two = [1, 0, 0, 0]

if system == 'straight_separatrix':
    correct_hom_uncertain = [1, 0, 0]
    correct_hom_zero = [1, 0, 0]
    correct_hom_one = [1, 0, 0]

if system == 'radial_2labels':
    correct_hom_uncertain = [1, 1, 0]
    correct_hom_zero = [1, 1, 0]
    correct_hom_one = [1, 0, 0]

if system == 'ellipsoidal_2d':
    correct_hom_uncertain = [1, 1, 0]
    correct_hom_zero = [1, 1, 0]
    correct_hom_one = [1, 0, 0]

if system == 'ellipsoidal_3d':
    correct_hom_uncertain = [1, 0, 1, 0]
    correct_hom_zero = [1, 0, 1, 0]
    correct_hom_one = [1, 0, 0, 0]

if system == 'ellipsoidal_4d' or system == 'ellipsoidal_larger_domain_4d' or system == 'ellipsoidal_larger_domain_4d_N16' or system == 'ellipsoidal_4d_more_balanced':
    correct_hom_uncertain = [1, 0, 0, 1, 0]
    correct_hom_zero = [1, 0, 0, 1, 0]
    correct_hom_one = [1, 0, 0, 0, 0]

if system == 'ellipsoidal_5d' or system == 'ellipsoidal_5d_larger_domain' or system == 'ellipsoidal_larger_domain_5d_N20' or system == 'ellipsoidal_larger_domain_5d_N15' or system == 'ellipsoidal_5d_data4' or system == 'ellipsoidal_5d_more_balanced_86_14':
    correct_hom_uncertain = [1, 0, 0, 0, 1, 0]
    correct_hom_zero = [1, 0, 0, 0, 1, 0]
    correct_hom_one = [1, 0, 0, 0, 0, 0]

if system == 'ellipsoidal_6d':
    correct_hom_uncertain = [1, 0, 0, 0, 0, 1, 0]
    correct_hom_zero = [1, 0, 0, 0, 0, 1, 0]
    correct_hom_one = [1, 0, 0, 0, 0, 0, 0]

if system == 'curved_separatrix':
    correct_hom_uncertain = [1, 0, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0, 0]

if system == 'radial_3labels':
    correct_hom_uncertain = [2, 2, 0]
    correct_hom_zero = [1, 1, 0]
    correct_hom_one = [1, 1, 0]
    correct_hom_two = [1, 0, 0]

def return_list_if_not_empty(row_elmt):
    if row_elmt == 'Empty region':
        return []
    elif row_elmt is None:
        return []
    else:
        return ast.literal_eval(row_elmt)

def check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one):
    checkpoint_u = False
    checkpoint_0 = False
    checkpoint_1 = False

    if correct_hom_uncertain == hom_uncertain:
        checkpoint_u = True
   #     print('checkpoint_u True')
    if correct_hom_zero == hom_zero:
        checkpoint_0 = True
   #     print('checkpoint_0 True')
    if correct_hom_one == hom_one:
        checkpoint_1 = True
   #     print('checkpoint_1 True')

 #   print('all: ', all([checkpoint_u, checkpoint_0, checkpoint_1]))
   # exit()
    return all([checkpoint_u, checkpoint_0, checkpoint_1])

def check_homology_4(system, correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, correct_hom_three, hom_uncertain, hom_zero, hom_one, hom_two, hom_three):
    checkpoint_u = False
    checkpoint_0 = False
    checkpoint_1 = False
    checkpoint_2 = False
    checkpoint_3 = False
    if correct_hom_uncertain == hom_uncertain:
        checkpoint_u = True
    if correct_hom_zero == hom_zero:
        checkpoint_0 = True
    if correct_hom_one == hom_one:
        checkpoint_1 = True
    if correct_hom_two == hom_two:
        checkpoint_2 = True
    if system == 'periodic' or system == 'periodic_balanced': # periodic_oversampled is intentionally not included here
        if hom_three == correct_hom_three or hom_three == []:
            checkpoint_3 = True
    elif correct_hom_three == hom_three:
        checkpoint_3 = True
    return all([checkpoint_u, checkpoint_0, checkpoint_1, checkpoint_2, checkpoint_3])

def check_homology_3(correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, hom_uncertain, hom_zero, hom_one, hom_two):
    checkpoint_u = False
    checkpoint_0 = False
    checkpoint_1 = False
    checkpoint_2 = False
    if correct_hom_uncertain == hom_uncertain:
        checkpoint_u = True
    if correct_hom_zero == hom_zero:
        checkpoint_0 = True
    if correct_hom_one == hom_one:
        checkpoint_1 = True
    if correct_hom_two == hom_two:
        checkpoint_2 = True
    return all([checkpoint_u, checkpoint_0, checkpoint_1, checkpoint_2])

if system == 'periodic' or system == 'periodic_balanced' or system == 'periodic_oversampled':
    possible_N_list = [15] #[12, 15, 18, 21, 24]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]

if system == 'M3D':
    possible_N_list = [12]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]

if system == 'straight_separatrix':
    possible_N_list = [2]
    #possible_N_list = [2, 4, 6, 8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
  #  possible_epsilon_list = [0.3]
if system == 'radial_3labels':
    possible_N_list = [10]
    #possible_N_list = [4, 8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'radial_2labels':
    possible_N_list = [4]
    #possible_N_list = [4, 8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'curved_separatrix':
    possible_N_list = [4]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'EMT' or system == 'EMT_threshold_pred' or system == 'EMT_corrected' or system == 'EMT_N6_1_28':
    possible_N_list = [6]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'ellipsoidal_2d':
    possible_N_list = [4] #[4, 8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49] #[0.1] #[0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'ellipsoidal_3d':
    possible_N_list = [6, 9, 12] #[6, 12]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49] #[0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'ellipsoidal_4d_more_balanced':
    possible_N_list = [8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'ellipsoidal_4d' or system == 'ellipsoidal_larger_domain_4d' or system == 'ellipsoidal_larger_domain_4d_N16':
    possible_N_list = [16]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49] #[0.3, 0.49] #[0.1, 0.2, 0.3, 0.4, 0.49]
if system ==  'ellipsoidal_larger_domain_5d_N15':
    possible_N_list = [15]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'ellipsoidal_5d' or system == 'ellipsoidal_5d_larger_domain' or system == 'ellipsoidal_larger_domain_5d' or system == 'ellipsoidal_larger_domain_5d_N20' or system == 'ellipsoidal_5d_data4' or system == 'ellipsoidal_5d_more_balanced_86_14':
    possible_N_list = [10]
    possible_epsilon_list = [0.2, 0.3, 0.4, 0.49] 
if system == 'ellipsoidal_6d':
    possible_N_list = [12]
    possible_epsilon_list = [0.3]

concatenated_results_filename = f'concatenated_results/{system}/{tmp_num}results-{system}.csv'
stat_file_name = f'concatenated_results/{system}/{tmp_num}stats-{system}.csv'
with open(stat_file_name, 'w', newline='') as writefile:
    writer2 = csv.writer(writefile)
    writer2.writerow(["N", "eps", "num_sucess_trials", "num_failed_trials", "mean_num_cubes_for_success", "SDEV_num_cubes_for_success"])
    for eps in possible_epsilon_list:
        print('Working on epsilon = ', eps)
        N_to_loss_fail_list_dict = dict()
        N_to_loss_suc_list_dict = dict()
        for N in possible_N_list:
            N_to_loss_fail_list_dict[N] = []
            N_to_loss_suc_list_dict[N] = []

        N_to_num_cubes_fail_dict = dict()
        N_to_num_cubes_success_dict = dict()
        for N in possible_N_list:
            N_to_num_cubes_fail_dict[N] = []
            N_to_num_cubes_success_dict[N] = []

        with open(concatenated_results_filename, 'w', newline='') as writefile:
            writer = csv.writer(writefile)
            writer.writerow(["ex_num", "N", "epsilon", "num_cubes", "final_test_loss", "result", "restart_count", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

            for job in range(num_jobs):
               # file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/{system}/{job}-results.csv'
                file_name = f'/Users/brittany/Documents/GitHub/attractor_identification_draft/output/results/{system}/{job}-results.csv'
               # file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/ellipsoidal_5d_more_balanced_86_14_{tmp_num}_{int(100-tmp_num)}/{job}-results.csv'
               # file_name = f'/Users/brittany/Documents/GitHub/attractor_identification_draft/output/results/ellipsoidal_5d_more_balanced_86_14_{tmp_num}_{int(100-tmp_num)}/{job}-results.csv'
                
                try:
                    with open(file_name, 'r') as file:
                        print('job: ', job)
                        reader = csv.DictReader(file)
                
                        # Iterate through each row in the CSV file
                        for row in reader:
                            # Access values by column name
                            ex_num = row['ex_num']
                            N = int(row['N'])
                            print("N", N)
                            optimizer_choice = row['optimizer_choice']
                            learning_rate = row['learning_rate']
                            epsilon = float(row['epsilon'])
                            num_cubes = int(row['num_cubes'])
                            final_test_loss = float(row['final_test_loss'])
                            restart_count = int(row['restart_count'])
                            hom_uncertain = return_list_if_not_empty(row['hom_uncertain'])
                            hom_zero = return_list_if_not_empty(row['hom_zero'])
                            hom_one = return_list_if_not_empty(row['hom_one'])
                            hom_two = return_list_if_not_empty(row['hom_two'])
                            hom_three = return_list_if_not_empty(row['hom_three'])

                            if system == 'periodic' or system == 'periodic_balanced' or system == 'periodic_oversampled':
                                v = check_homology_4(system, correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, correct_hom_three, hom_uncertain, hom_zero, hom_one, hom_two, hom_three)
                            elif system == 'straight_separatrix':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'radial_2labels':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'curved_separatrix':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'EMT' or system == 'EMT_corrected':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'ellipsoidal_2d':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'radial_3labels':
                                v = check_homology_3(correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, hom_uncertain, hom_zero, hom_one, hom_two)
                            elif system == 'M3D':
                                v = check_homology_3(correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, hom_uncertain, hom_zero, hom_one, hom_two)    
                            else:
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                        
                            if system == 'radial_3labels':
                                if math.isclose(epsilon, eps) and final_test_loss < 0.15:
                                    if v:
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                    else:
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                            elif system == 'periodic':
                                if math.isclose(epsilon, eps) and final_test_loss < 0.49:
                                    if v:
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                    else:
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                            elif system == 'periodic_balanced':
                                if math.isclose(epsilon, eps) and final_test_loss < 1:
                                    if v:
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                    else:
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                            elif system == 'M3D':
                                if math.isclose(epsilon, eps) and final_test_loss < 0.27:
                                    if v:
                                        print('success')
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                    else:
                                        print('failure')
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                            elif system == 'ellipsoidal_5d_more_balanced_86_14':
                                if math.isclose(epsilon, eps) and final_test_loss < 0.11:
                                    if v:
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                    else:
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                            #elif system == 'curved_separatrix':
                            #    if math.isclose(epsilon, eps) and final_test_loss < 0.06:
                            #        if v:
                            #            N_to_loss_suc_list_dict[N].append(final_test_loss)
                            #            N_to_num_cubes_success_dict[N].append(num_cubes)
                            #        else:
                            #            N_to_loss_fail_list_dict[N].append(final_test_loss)
                            #            N_to_num_cubes_fail_dict[N].append(num_cubes)
                            else:
                                if math.isclose(epsilon, eps):
                                    if v:
                                        N_to_loss_suc_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_success_dict[N].append(num_cubes)
                                        print('success')
                                    else:
                                        N_to_loss_fail_list_dict[N].append(final_test_loss)
                                        N_to_num_cubes_fail_dict[N].append(num_cubes)
                                        print('failure')

                            writer.writerow([ex_num, N, epsilon, num_cubes, final_test_loss, v, restart_count, hom_uncertain, hom_zero, hom_one, hom_two, hom_three])
                                
                except:
                    print('No results for job # ', job)

        for N in possible_N_list:
            print('N: ', N)
            loss_failure_list = N_to_loss_fail_list_dict[N]
            loss_success_list = N_to_loss_suc_list_dict[N]
            cubes_success_list = N_to_num_cubes_success_dict[N]
            cubes_failure_list = N_to_num_cubes_fail_dict[N]

          #  print(loss_failure_list)
           # print(loss_success_list)

            # Combine both lists with labels (False for failure, True for success)
            combined = [(num, False) for num in loss_failure_list] + [(num, True) for num in loss_success_list]
            #print('combined: ', combined)

            # Sort by the number
            combined.sort()

            # Extract sorted numbers and corresponding bools
            sorted_numbers, sorted_bools = zip(*combined)

            # Convert to lists
            sorted_numbers = list(sorted_numbers)
            sorted_bools = list(sorted_bools)

            file_path3 = f'concatenated_results/{system}/scatter_{N}_epsilon_{eps*10}.png'

            plot_results(sorted_numbers, sorted_bools, file_path3)


            print('len of failure: ', len(loss_failure_list))
            print('len of success: ', len(loss_success_list))
            if len(cubes_success_list) != 0:
                mean_raw = sum(cubes_success_list)/len(cubes_success_list)
                mean = round(mean_raw, 1)
                print('mean number of successful cubes: ', mean)
                variance = sum([((x - mean) ** 2) for x in cubes_success_list]) / len(cubes_success_list)
                res = variance ** 0.5
                SDEV = round(res, 1)
                print('SDEV: ', SDEV)
                

                file_path1 = f'concatenated_results/{system}/histogram_{N}_epsilon_{eps*10}'
                file_path2 = f'concatenated_results/{system}/cube_dist_histogram_{N}_epsilon_{eps*10}'
            
                # plot_loss_histogram(loss_success_list, loss_failure_list, 'test', file_path1, system, N, 'Test Loss')
                # plot_loss_histogram(cubes_success_list, cubes_failure_list, 'test', file_path2, system, N, 'Number of Cubes')
            else:
                mean = 'None'
                SDEV = 'None'
            writer2.writerow([N, eps, len(cubes_success_list), len(cubes_failure_list), mean, SDEV])
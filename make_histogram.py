import csv
from src.attractor_id.histograms import plot_loss_histogram
import ast
import math

num_jobs = 250
system = 'EMT'

if system == 'EMT':
    correct_hom_uncertain = [1, 0, 0, 0, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0, 0, 0, 0]

if system == 'periodic':
    correct_hom_uncertain = [1, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0]
    correct_hom_two = [1, 0, 0, 0]
    correct_hom_three = [1, 0, 0, 0]

if system == 'straight_separatrix':
    correct_hom_uncertain = [1, 0, 0]
    correct_hom_zero = [1, 0, 0]
    correct_hom_one = [1, 0, 0]

if system == 'radial_2labels':
    correct_hom_uncertain = [1, 1, 0]
    correct_hom_zero = [1, 1, 0]
    correct_hom_one = [1, 0, 0]

if system == 'curved_separatrix':
    correct_hom_uncertain = [1, 0, 0, 0, 0]
    correct_hom_zero = [1, 0, 0, 0, 0]
    correct_hom_one = [1, 0, 0, 0, 0]

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
    if correct_hom_zero == hom_zero:
        checkpoint_0 = True
    if correct_hom_one == hom_one:
        checkpoint_1 = True
    return all([checkpoint_u, checkpoint_0, checkpoint_1])

def check_homology_3(correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, correct_hom_three, hom_uncertain, hom_zero, hom_one, hom_two, hom_three):
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
    if correct_hom_three == hom_three:
        checkpoint_3 = True
    return all([checkpoint_u, checkpoint_0, checkpoint_1, checkpoint_2, checkpoint_3])

concatenated_results_filename = f'concatenated_results/results-{system}.csv'

if system == 'periodic':
    possible_N_list = [12, 15, 18, 21, 24]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'straight_separatrix':
    possible_N_list = [2, 4, 6, 8]
    possible_epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.49]
if system == 'radial_2labels':
    possible_N_list = [4, 8]
    possible_epsilon_list = [0.3]
if system == 'curved_separatrix':
    possible_N_list = [4]
    possible_epsilon_list = [0.3]
if system == 'EMT':
    possible_N_list = [18]
    possible_epsilon_list = [0.1, 0.3, 0.49]


for eps in possible_epsilon_list:
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
        writer.writerow(["ex_num", "N", "epsilon", "num_cubes", "final_test_loss", "result", "hom_uncertain", "hom_zero", "hom_one", "hom_two", "hom_three"])

        for job in range(num_jobs):
            if system == 'periodic':
                file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/{system}/{job}-results.csv'
            elif system == 'straight_separatrix':
                file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/system1_old/{job}-results.csv'
            elif system == 'radial_2labels':
                file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/{system}/{job}-results.csv'
            elif system == 'curved_separatrix':
                file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/{system}/{job}-results.csv'
            elif system == 'EMT':
                file_name = f'/home/bg545/attractor_id/attractor_identification_draft/output/results/{system}/{job}-results.csv'
            #print('file name: ', file_name)
            try:
                with open(file_name, 'r') as file:
                    reader = csv.DictReader(file)
            
                    # Iterate through each row in the CSV file
                    for row in reader:
                        # Access values by column name
                        ex_num = row['ex_num']
                        N = int(row['N'])
                        optimizer_choice = row['optimizer_choice']
                        learning_rate = row['learning_rate']
                        epsilon = float(row['epsilon'])
                        num_cubes = int(row['num_cubes'])
                        final_test_loss = float(row['final_test_loss'])
                        hom_uncertain = return_list_if_not_empty(row['hom_uncertain'])
                        hom_zero = return_list_if_not_empty(row['hom_zero'])
                        hom_one = return_list_if_not_empty(row['hom_one'])
                        hom_two = return_list_if_not_empty(row['hom_two'])
                        hom_three = return_list_if_not_empty(row['hom_three'])

                        if math.isclose(epsilon, eps):
                            if system == 'periodic':
                                v = check_homology_3(correct_hom_uncertain, correct_hom_zero, correct_hom_one, correct_hom_two, correct_hom_three, hom_uncertain, hom_zero, hom_one, hom_two, hom_three)
                            elif system == 'straight_separatrix':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'radial_2labels':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'curved_separatrix':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            elif system == 'EMT':
                                v = check_homology_2(correct_hom_uncertain, correct_hom_zero, correct_hom_one, hom_uncertain, hom_zero, hom_one)
                            
                            else:
                                print('Not implemented')
                                exit()

                            if v:
                                N_to_loss_suc_list_dict[N].append(final_test_loss)
                                N_to_num_cubes_success_dict[N].append(num_cubes)
                            else:
                                N_to_loss_fail_list_dict[N].append(final_test_loss)
                                N_to_num_cubes_fail_dict[N].append(num_cubes)

                        writer.writerow([ex_num, N, epsilon, num_cubes, final_test_loss, v, hom_uncertain, hom_zero, hom_one, hom_two, hom_three])
                                
            
            except:
                pass
                print('No results for job number ', job)

    for N in possible_N_list:
        print('N: ', N)
        loss_failure_list = N_to_loss_fail_list_dict[N]
        loss_success_list = N_to_loss_suc_list_dict[N]
        cubes_success_list = N_to_num_cubes_success_dict[N]
        cubes_failure_list = N_to_num_cubes_fail_dict[N]
        print('len of failure: ', len(loss_failure_list))
        print('len of success: ', len(loss_success_list))
        file_path1 = f'concatenated_results/{system}/histogram_{N}_epsilon_{eps*10}'
        file_path2 = f'concatenated_results/{system}/cube_dist_histogram_{N}_epsilon_{eps*10}'
        plot_loss_histogram(loss_success_list, loss_failure_list, 'test', file_path1, system, N, 'Loss')
        plot_loss_histogram(cubes_success_list, cubes_failure_list, 'test', file_path2, system, N, 'Number of Cubes')
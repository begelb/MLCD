from copy import deepcopy
from itertools import combinations

''' Compute all possible binary lists of length d '''
def make_all_binary_lists(length):
    # base case for recursive definition
    if length == 1:
        return [[0], [1]]
    else:
        list_d_minus_one = make_all_binary_lists(length - 1)
        list_d_minus_one_copy = deepcopy(list_d_minus_one)

        # for each list-element of these lists, insert 0 or 1 respectively at the end of the list-element
        for i in range(0, len(list_d_minus_one)):
            list_d_minus_one[i].insert(0, 0)
            list_d_minus_one_copy[i].insert(0, 1)

        # return the concatenation of the modified lists
        return list_d_minus_one + list_d_minus_one_copy
    
def get_combinations_minus_one_element(my_list):
    j = len(my_list) - 1
    return list(combinations(my_list, j))

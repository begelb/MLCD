''' This file contains the Config class definition. It has an init function that reads the configuration variables from the configuration file given a file name (path). It also has a check_types function that checks whether the configuration variables are of the type expected by the program. This is intended to catch typos in the configuration file and exit the program if the the configuration variables have the wrong type.

Acknowledgement: This configuration set-up is based on the GitHub repository "MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space" written by Ewerton Vieira, Aravind Sivaramakrishnan, and Sumanth Tangirala (2023) and available under MIT License at https://github.com/Ewerton-Vieira/MORALS/tree/main

'''

import ast
from sys import exit

class Config:

    def __init__(self, config_fname):
        with open(config_fname) as f:
            config = eval(f.read())
        self.example_type = config["example_type"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.optimizer_choice = config["optimizer_choice"]
        self.verbose = bool(config["verbose"])
        self.make_figures = bool(config["make_figures"])
        self.dimension = config["dimension"]
        self.data_bounds = ast.literal_eval(config["data_bounds"])
        self.train_data_file = config["train_data_file"]
        self.test_data_file = config["test_data_file"]
        self.train_url = config["train_url"]
        self.test_url = config["test_url"]
        self.num_labels = config["num_labels"]
        self.N_list = ast.literal_eval(config["network_width_list_for_experiment"])
        self.using_pandas = not bool(config["using_local_data"])
        self.results_directory = config["results_directory"]
        self.models_directory = config["models_directory"]
        self.figures_directory = config["figures_directory"]
        self.threshold_prediction = config["threshold_prediction"]

    def check_types(self):
        if type(self.example_type) is not str:
            raise Exception("Example type has the incorrect type. Must be " + str(str) + ". Found a " + str(type(self.example_type)))
        if type(self.learning_rate) is not float:
            raise Exception("Learning rate has the incorrect type. Must be " + str(float) + ". Found a " + str(type(self.learning_rate)))
        if type(self.epochs) is not int:
            raise Exception("Epochs has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.epochs)))
        if type(self.optimizer_choice) is not str:
            raise Exception("Optimizer choice has the incorrect type. Must be " + str(str) +". Found a " + str(type(self.optimizer_choice)))
        if type(self.verbose) is not bool:
            raise Exception("Verbose has the incorrect type. Must be " + str(bool) +". Found a" + str(type(self.verbose)))
        if type(self.make_figures) is not bool:
            raise Exception("Make figures has the incorrect type. Must be " + str(str) + ". Found a" + str(type(self.make_figures)))
        if type(self.dimension) is not int:
            raise Exception("Dimension has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.dimension)))
        if type(self.data_bounds) is not list:
            raise Exception("Data bounds has the incorrect type. Must be " + str(list) + ". Found a " + str(type(self.data_bounds)))
        for bound in self.data_bounds:
            if type(bound) is not float and type(bound) is not int:
                raise Exception("Data bounds list element has the incorrect type. Must be " + str(float) + " or " + str(int) + ". Found a " + str(type(bound)))
        if type(self.train_data_file) is not str:
            raise Exception("Train data file has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.train_data_file)))
        if type(self.test_data_file) is not str:
            raise Exception("Test data file has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.test_data_file)))
        if type(self.train_url) is not str:
            raise Exception("Train URL has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.train_url)))
        if type(self.test_url) is not str:
            raise Exception("Test URL has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.test_url)))
        if type(self.num_labels) is not int:
            raise Exception("Number of labels has the wrong type. Must be " + str(int) + ". Found a " + str(type(self.num_labels)))
        if type(self.N_list) is not list:
            raise Exception("Network_width_list_for_experiment has the wrong type. Must be " + str(list) + ". Found a " + str(type(self.N_list)))
        for N in self.N_list:
            if type(N) is not int:
                raise Exception("Network width inside network_width_list_for_experiment has the wrong type. Must be " + str(int) + ". Found a " + str(type(N)))
        if type(self.using_pandas) is not bool:
            raise Exception("using_local_data has the wrong type. Must be " + str(bool) + ". Found a " + str(type(self.using_pandas)))
        if type(self.results_directory) is not str:
            raise Exception("Results directory has the wrong type. Must be " + str(str) + ". Found a " + str(type(self.results_directory)))
        if type(self.models_directory) is not str:
            raise Exception("Models directory has the wrong type. Must be " + str(str) + ". Found a " + str(type(self.models_directory)))
        if type(self.figures_directory) is not str:
            raise Exception("Figures directory has the wrong type. Must be " + str(str) + ". Found a " + str(type(self.figures_directory)))
        if type(self.threshold_prediction) is not bool:
            raise Exception("Threshold prediction has the wrong type. Must be " + str(bool) + ". Found a " + str(type(self.threshold_prediction)))


def user_warning_about_N_and_dimension(config, N):
    if N % config.dimension != 0:
        print("Warning: N is expected to be an integer multiple of the dimension of the system, which is " + str(config.dimension))
        user_choice = input("Do you want to continue? (y/n):")
        if user_choice != "y":
            raise Exception("User terminated program.")

def configure(config_fname):
    config = Config(config_fname)
    config.check_types()
    return config
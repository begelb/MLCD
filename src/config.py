''' This file contains the Config class definition. It has an init function that reads the configuration variables from the configuration file given a file name (path). It also has a check_types function that checks whether the configuration variables are of the type expected by the program. This is intended to catch typos in the configuration file and exit the program if the the configuration variables have the wrong type.

Acknowledgement: This configuration set-up is based on the GitHub repository "MORALS: Morse Graph-aided discovery of Regions of Attraction in a learned Latent Space" written by Ewerton Vieira, Aravind Sivaramakrishnan, and Sumanth Tangirala (2023) and available under MIT License at https://github.com/Ewerton-Vieira/MORALS/tree/main

'''

import ast

class Config:

    def __init__(self, config_fname):
        with open(config_fname) as f:
            config = eval(f.read())
        self.system = config["system"]
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
        self.num_labels = config["num_labels"]
        self.N_list = ast.literal_eval(config["network_width_list_for_experiment"])

    def check_types(self):
        if type(self.system) is not int:
            print("System has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.system)))
            exit()
        if type(self.example_type) is not str:
            print("Example type has the incorrect type. Must be " + str(str) + ". Found a " + str(type(self.example_type)))
            exit()
        if type(self.learning_rate) is not float:
            print("Learning rate has the incorrect type. Must be " + str(float) + ". Found a " + str(type(self.learning_rate)))
            exit()
        if type(self.epochs) is not int:
            print("Epochs has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.epochs)))
            exit()
        if type(self.optimizer_choice) is not str:
            print("Optimizer choice has the incorrect type. Must be " + str(str) +". Found a " + str(type(self.optimizer_choice)))
            exit()
        if type(self.verbose) is not bool:
            print("Verbose has the incorrect type. Must be " + str(bool) +". Found a" + str(type(self.verbose)))
            exit()
        if type(self.make_figures) is not bool:
            print("Make figures has the incorrect type. Must be " + str(str) + ". Found a" + str(type(self.make_figures)))
            exit()
        if type(self.dimension) is not int:
            print("Dimension has the incorrect type. Must be " + str(int) + ". Found a " + str(type(self.dimension)))
            exit()
        if type(self.data_bounds) is not list:
            print("Data bounds has the incorrect type. Must be " + str(list) + ". Found a " + str(type(self.data_bounds)))
            exit()
        for bound in self.data_bounds:
            if type(bound) is not float and type(bound) is not int:
                print("Data bounds list element has the incorrect type. Must be " + str(float) + " or " + str(int) + ". Found a " + str(type(bound)))
                exit()
        if type(self.train_data_file) is not str:
            print("Train data file has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.train_data_file)))
            exit()
        if type(self.test_data_file) is not str:
            print("Test data file has the incorrect type. Must be " + str(str) + " Found a " + str(type(self.test_data_file)))
            exit()
        if type(self.num_labels) is not int:
            print("Number of labels has the wrong type. Must be " + str(int) + ". Found a " + str(type(self.num_labels)))
            exit()
        if type(self.N_list) is not list:
            print("Network_width_list_for_experiment has the wrong type. Must be " + str(list) + ". Found a " + str(type(self.N_list)))
            exit()
        for N in self.N_list:
            if type(N) is not int:
                print("Network width inside network_width_list_for_experiment has the wrong type. Must be " + str(int) + ". Found a " + str(type(N)))
                exit()      

def user_warning_about_N_and_dimension(config, N):
    if N % config.dimension != 0:
        print("Warning: N is expected to be an integer multiple of the dimension of the system, which is " + str(config.dimension))
        user_choice = input("Do you want to continue? (y/n):")
        if user_choice != "y":
            exit()  

def configure(config_fname):
    config = Config(config_fname)
    config.check_types()
    return config
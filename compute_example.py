from utils.experiment_utils import compute_example
from utils.config_utils import configure

'''' Global variables set by user '''

system = 1
N = 10
labeling_threshold = 0.3
job_index = 0

def main():
    config_fname = f'config/system{system}.txt'
    config = configure(config_fname)
    compute_example(config, job_index, N, labeling_threshold)

if __name__ == "__main__":
    main()
    
# job_index = int(sys.argv[1])

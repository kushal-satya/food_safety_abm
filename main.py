# Import required libraries
import numpy as np
from tqdm import tqdm
import concurrent.futures
import numba as nb
from matplotlib import pyplot as plt

# Import required files
import config 

from src.simulation import* 
from src.data import*

# Set the number of iterations
iterations = 1
f_test_array = np.arange(0.01, 0.05, 0.01)
p_test_array = np.arange(0.01, 0.05, 0.01)

if __name__ == '__main__':
    iterations = 1
    ftest_rate = []
    ptest_rate = []
    costs_mean = []
    costs_std = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for f_test_rate in tqdm(f_test_array, desc='f_test_rate'):
            futures.extend(
                executor.submit(run_test_wrapper, f_test_rate, p_test_rate, iterations)
                for p_test_rate in p_test_array
            )
        for future in concurrent.futures.as_completed(futures):
            f_test_rate, p_test_rate, cost_mean, cost_std = future.result()
            ftest_rate.append(f_test_rate)
            ptest_rate.append(p_test_rate)
            costs_mean.append(cost_mean)
            costs_std.append(cost_std)

    lists_all = [ftest_rate, ptest_rate, costs_mean, costs_std]
    file_names = ['ftest_rate', 'ptest_rate', 'costs_mean', 'costs_std']
    print('done!!')
    for list, file_name in zip(lists_all, file_names):
        pickle_dump(file_name, list)


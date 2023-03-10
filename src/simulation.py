import concurrent.futures
import numpy as np
from tqdm import tqdm

from src.costs import*
from src.testing import*


def run_test_wrapper(f_test_rate, p_test_rate, iterations):
    costs = []
    for _ in range(iterations):
        total_cost = run_test(f_test_rate, p_test_rate)
        costs.append(total_cost)
    return f_test_rate, p_test_rate, np.mean(costs), np.std(costs)
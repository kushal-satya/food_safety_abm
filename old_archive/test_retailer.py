################################
####### RETAILER TEST #############
################################
import numpy as np 
import pandas as pd

r_test_number = np.random.rand(number_box, 1)
r_test_number = r_test_number < r_test_rate

r_test_history_box_id_cusult[:, iteration] = (r_test_number == 1 & contaminate_b[:, iteration] == 1 & contaminate_drop[:, iteration] == 0)
r_test_location = np.where(r_test_number == 1 & contaminate_b[:, iteration] == 1 & contaminate_drop[:, iteration] == 0)

r_test_number = (r_test_number == 1 & contaminate_drop[:, iteration] == 0)
# How many boxes are tested in this process
r_test_location_size = r_test_location.shape[0]

# Drop contaminate box in same plot
drop_plot_r = (r_test_location - 1) // box_per_plot + 1

for i in range(number_box):
    for j in range(r_test_location_size):
        if (i - 1) // box_per_plot == (r_test_location[j] - 1) // box_per_plot:
            contaminate_drop[i, iteration] = 1

# How many boxes will be delivered from retailer to customer
r_to_c_box = (contaminate_drop[:, iteration] == 0)

# Add customer ID
cur_box = 0
cur_history_box_id_cus = 0
used_pound = 0

for i in range(len(customer_demand)):
    while cur_box < len(history_box_id_r) and r_to_c_box[cur_box] == 0:
        history_box_id_cus[cur_history_box_id_cus, iteration] = history_box_id_r[cur_box, iteration] * 10000000
        cur_box += 1
        cur_history_box_id_cus += 1
    if cur_box >= len(history_box_id_r):
        break
    cur_need = customer_demand[i]
    while cur_need > 0 and r_to_c_box[cur_box] == 1 and cur_box < len(history_box_id_r):
        if used_pound + cur_need < box_cap:
            used_pound += cur_need
            cur_need = 0
            history_box_id_cus[cur_history_box_id_cus, iteration] = history_box_id_r[cur_box, iteration] * 10000000 + i
            cur_history_box_id_cus += 1
        else:
            cur_need -= (box_cap - used_pound)
            used_pound = 0
            history_box_id_cus[cur_history_box_id_cus, iteration] = history_box_id_r[cur_box, iteration] * 10000000 + i
            cur_history_box_id_cus += 1
            cur_box += 1

def run_stage(box_ids, contamination_mask, test_rate):
    tested_boxes_mask = np.random.rand(box_ids.shape[0]) < test_rate
    tested_contaminated_boxes = box_ids[np.logical_and(contamination_mask, tested_boxes_mask)]
    if tested_contaminated_boxes.size > 0:
        dropped_boxes = set()
        farm_idxs = tested_contaminated_boxes // 10000000000
        plot_idxs = (tested_contaminated_boxes % 10000000000) // 100000
        plot_boxes = np.array([generate_box_id(farm_idxs[i], plot_idxs[i], box_idx) for i in range(tested_contaminated_boxes.size) for box_idx in range(1, box_per_plot+1)])
        dropped_boxes.update(plot_boxes)
        remaining_boxes = box_ids[~isin(box_ids, dropped_boxes)]
    else:
        dropped_boxes = set()
        remaining_boxes = box_ids
    return remaining_boxes, contamination_mask, dropped_boxes

farm_idxs = tested_contaminated_boxes // 100
plot_idxs = (tested_contaminated_boxes % 100) // 100000
plot_boxes = np.array([generate_box_id(farm_idxs[i], plot_idxs[i], box_idx) for i in range(tested_contaminated_boxes.size) for box_idx in range(1,box_ids.shape[0]+1 )])
dropped_boxes.update(plot_boxes)

def generate_box_id(farm_idx, plot_idx, box_idx):
        return farm_idx * 100 + plot_idx * 10 + box_idx * 1

generate_box_id(1,2,3)
farm_idxs = tested_contaminated_boxes // 100
plot_idxs = (tested_contaminated_boxes % 100) // 100000
print(farm_idxs)
print(plot_idxs)


import timeit
import numpy as np

def generate_box_id(farm_idx, plot_idx, box_idx):
    return farm_idx * 10000000000 + plot_idx * 100000 + box_idx * 10

def generate_box_ids_list(farm_population, plot_per_farm, box_per_plot):
    box_ids = [generate_box_id(farm_idx, plot_idx, box_idx)
               for farm_idx in range(1, farm_population + 1)
               for plot_idx in range(1, plot_per_farm + 1)
               for box_idx in range(1, box_per_plot + 1)]
    return box_ids

def generate_box_ids_numpy(farm_population, plot_per_farm, box_per_plot):
    farm_range = np.arange(1, farm_population + 1)
    plot_range = np.arange(1, plot_per_farm + 1)
    box_range = np.arange(1, box_per_plot + 1)
    farm_grid, plot_grid, box_grid = np.meshgrid(farm_range, plot_range, box_range)
    box_ids = (farm_grid * 10000000000 + plot_grid * 100000 + box_grid * 10).ravel()
    return box_ids

farm_population = 100
plot_per_farm = 10
box_per_plot = 1000

# List comprehension
start_time = timeit.default_timer()
box_ids_list = generate_box_ids_list(farm_population, plot_per_farm, box_per_plot)
elapsed_time_list = timeit.default_timer() - start_time
print(f"List comprehension: {elapsed_time_list:.6f} seconds")

# Numpy array
start_time = timeit.default_timer()
box_ids_numpy = generate_box_ids_numpy(farm_population, plot_per_farm, box_per_plot)
elapsed_time_numpy = timeit.default_timer() - start_time
print(f"Numpy array: {elapsed_time_numpy:.6f} seconds")

print(f"Numpy array is {elapsed_time_list / elapsed_time_numpy:.1f}x faster")

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

import numpy as np 

farm_population = 1
packer_population = 1
distributor_population = 1
retailer_population = 1
customer_population = 5000

plot_per_farm = 1856
box_per_P = 31562
box_per_D = 31562
box_per_R = 31562

box_per_plot = 17
contamination_rate = 0.06

d_test_rate = 0.0
r_test_rate = 0.0

box_ids_l=[]
box_ids_C_l=[]
dropped_boxes_F_l=[]
dropped_boxes_P_l=[]
dropped_boxes_D_l=[]
dropped_boxes_R_l=[]
dropped_boxes_C_l=[]
detection_percent_l=[]

iterations = 10

# Test cost at each stage
test_cost_F = 350
test_cost_P = 50
test_cost_D = 50
test_cost_R = 50

# Customer demand
box_cap = 50
number_plot = farm_population * plot_per_farm
number_box = number_plot * box_per_plot
customer_number = int (number_box * box_cap * (0.80 - contamination_rate))
customer_demand = np.floor(np.abs(np.random.normal(1, 2, (customer_number, 1))))


# Customer illness cost
ill_rate = 0.04
hospital_rate = 0.0164
death_rate = 0.000041
ill_compensation = 719
hospital_compensation = 18438
death_compensation = 1764112

# Recall and Trace cost
unit_recall_labor_cost = 10
unit_trace_labor_cost = 10
price_per_box = 100

# Transportation cost
unit_trans_cost = 0.007
cost_indicator = 1
F_P_distance = 5
P_D_distance = 2983
D_R_distance = 11
# Cost Calculation 

import numpy as np
import config
box_per_P = config.box_per_P
box_per_D = config.box_per_D
box_per_R = config.box_per_R
test_cost_F = config.test_cost_F
test_cost_P = config.test_cost_P
test_cost_D = config.test_cost_D
test_cost_R = config.test_cost_R
F_P_distance = config.F_P_distance
P_D_distance = config.P_D_distance
D_R_distance = config.D_R_distance
unit_trans_cost = config.unit_trans_cost
customer_number = config.customer_number
customer_demand = config.customer_demand
ill_rate = config.ill_rate
hospital_rate = config.hospital_rate
death_rate = config.death_rate
ill_compensation = config.ill_compensation
hospital_compensation = config.hospital_compensation
death_compensation = config.death_compensation
unit_recall_labor_cost = config.unit_recall_labor_cost
unit_trace_labor_cost = config.unit_trace_labor_cost
price_per_box = config.price_per_box
box_cap = config.box_cap


# Transportation cost
def transportation_cost(box_ids_F_P,box_ids_P_D,box_ids_D_R):
    trans_cost_F_P = (len(box_ids_F_P) // box_per_P +1 )* F_P_distance * unit_trans_cost
    trans_cost_P_D = (len(box_ids_P_D) // box_per_D +1 )* P_D_distance * unit_trans_cost
    trans_cost_D_R = (len(box_ids_D_R) // box_per_R + 1) * D_R_distance * unit_trans_cost
    return trans_cost_D_R+trans_cost_P_D+trans_cost_F_P

def testing_cost(tests_F,tests_P,tests_D,tests_R):
    return tests_F*test_cost_F+tests_P*test_cost_P+tests_D*test_cost_D+tests_R*test_cost_R

# Customer consumption and illness report 
def customer_illness_cost(box_ids_R_C,contamination_mask):
    #boxes_allotted = np.zeros((customer_number,2),dtype=int)
    all_boxes= box_ids_R_C
    boxes_allotted_cont = np.zeros(customer_number,dtype=bool)
    current_box_cap = box_cap
    current_box_id = 0
    current_box_contaminated = contamination_mask[current_box_id]
    if contaminated := sum(contamination_mask):
        for i, current_customer_demand in enumerate(customer_demand):
            if current_customer_demand == 0:
                continue
            if current_box_cap - current_customer_demand >= 0:
                #boxes_allotted[i,0]=all_boxes[current_box_id]
                current_box_cap -= current_customer_demand
                boxes_allotted_cont[i]=current_box_contaminated
                if current_box_cap == 0:
                    current_box_id += 1
                    current_box_contaminated = (
                        contamination_mask[current_box_id]
                        if current_box_id < len(all_boxes)
                        else False
                    )
                    current_box_cap = box_cap
            else: 
                #boxes_allotted[i,0]=all_boxes[current_box_id]
                current_box_id += 1
                if current_box_id < len(all_boxes):
                    cont= contamination_mask[current_box_id]
                else:
                    cont=current_box_contaminated
                current_box_cap = box_cap - current_customer_demand - current_box_cap
                boxes_allotted_cont[i]=cont+current_box_contaminated 
                current_box_contaminated = cont
                #boxes_allotted[i,1]=all_boxes[current_box_id]
        customers_contaminated = sum(boxes_allotted_cont)
    else:
        customers_contaminated = 0

    ill_number = np.random.rand(customers_contaminated, 1)
    hospital_number = np.random.rand(customers_contaminated, 1)
    death_number = np.random.rand(customers_contaminated, 1)

    death_number = death_number < death_rate
    death_case_number = np.sum(death_number)

    hospital_number = hospital_number < hospital_rate
    hospital_case_number = np.sum(hospital_number)

    ill_number = ill_number < ill_rate
    ill_case_number = np.sum(ill_number)
    return (
        death_case_number * death_compensation
        + hospital_case_number * hospital_compensation
        + ill_case_number * ill_compensation
    )

def recall_cost(contamination_mask):
    return sum(contamination_mask)* (unit_recall_labor_cost+unit_trace_labor_cost+price_per_box)
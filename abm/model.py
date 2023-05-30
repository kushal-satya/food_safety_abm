import numpy as np 
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from supply import * 
from monthly import *
from supply import *

import gc
gc.enable()
class Run(object):
    def __init__(self,parameters):
        self.parameters = parameters
    
    def initiate_farms(self):
        farms = []
    
    def initiate_supply_chain(self):
        self.supply = Supply(self.parameters)     
        
class Farms(object):
    def __init__(self,parameters):
        self.parameters = parameters
        self.farms = []
        self.farm_ids = []
        self.total_boxes = parameters.farm_population * parameters.plot_per_farm * parameters.box_per_plot
        self.box_ids = self.generate_box_ids_numpy(parameters.farm_population, parameters.plot_per_farm, parameters.box_per_plot)
        
    
    def find_optimal_testing_rate(self, parameters):
        for f_test_rate in parameters.f_test_range:
            self.parameters.f_test_rate = f_test_rate
            for p_test_rate in parameters.p_test_range:
                self.parameters.p_test_rate = p_test_rate
                for d_test_rate in parameters.d_test_range:
                    self.parameters.d_test_rate = d_test_rate
                    for r_test_rate in parameters.r_test_range:
                        self.parameters.r_test_rate = r_test_rate
                        self.run_stage(parameters)
        
           
    # generate contaminated box mask
    def generate_contamination_mask(self):
        contaminated_boxes = int(self.parameters.total_boxes * self.parameters.contamination_rate)
        mask = np.zeros(self.parameters.total_boxes, dtype=np.bool_)
        mask[:contaminated_boxes] = True
        np.random.shuffle(mask)
        return mask

    # generate box id
    def generate_box_ids_numpy(self, parameters):
        farm_population = parameters.farm_population
        plot_per_farm = parameters.plot_per_farm
        box_per_plot = parameters.box_per_plot
        farm_range = np.arange(1, farm_population + 1)
        plot_range = np.arange(1, plot_per_farm + 1)
        box_range = np.arange(1, box_per_plot + 1)
        farm_grid, plot_grid, box_grid = np.meshgrid(farm_range, plot_range, box_range)
        return (farm_grid * 10000000000 + plot_grid * 100000 + box_grid).ravel()       
    
    def run_stage(self, parameters):
        tested_boxes_mask = np.random.rand(self.box_ids.shape[0]) < parameters.test_rate
        tested_contaminated_boxes = self.box_ids[np.logical_and(contamination_mask, tested_boxes_mask)]
        
        if tested_contaminated_boxes.size > 0:
            plot_farm_ids_tested = tested_contaminated_boxes - tested_contaminated_boxes % 100
            #plot_farm_ids_all = box_ids - box_ids % 100
            tested_contaminated_boxes_all=np.repeat(np.unique(plot_farm_ids_tested),box_per_plot)+ np.tile(np.arange(1, box_per_plot + 1),np.unique(plot_farm_ids_tested).shape[0])
            #plot_boxes = np.array([generate_box_id(farm_idxs[i], plot_idxs[i], box_idx) for i in range(tested_contaminated_boxes.size) for box_idx in range(1, box_per_plot+1)])
            #remaining_boxes = box_ids[~np.isin(box_ids, dropped_boxes)]
            #tested_contaminated_boxes= np.array(list(dropped_boxes))
            dropped_boxes = set(np.unique(tested_contaminated_boxes_all))
            mask = np.isin(self.box_ids, tested_contaminated_boxes_all)
            contamination_mask = contamination_mask[~mask]
            box_ids_n = self.box_ids[~mask]
            #dropped_boxes = set(tested_contaminated_boxes)
        else:
            dropped_boxes = set()
            box_ids_n = self.box_ids
        remaining_boxes = box_ids_n
        return remaining_boxes, contamination_mask, dropped_boxes, sum(tested_boxes_mask)

    # Run testing and return the number of boxes dropped at each stage
    def run_test(self, parameters):
        
        contamination_mask = self.generate_contamination_mask(self.total_boxes, self.contamination_rate)
        box_ids_F_P, contamination_mask1, dropped_boxes_F,tests_F = self.run_stage(box_ids, contamination_mask, self.f_test_rate)
        box_ids_P_D, contamination_mask2, dropped_boxes_P,tests_P = self.run_stage(box_ids_F_P, contamination_mask1, p_test_rate)
        box_ids_D_R, contamination_mask3, dropped_boxes_D,tests_D = self.run_stage(box_ids_P_D, contamination_mask2, d_test_rate)
        box_ids_R_C, contamination_mask4, dropped_boxes_R,tests_R = self.run_stage(box_ids_D_R, contamination_mask3, r_test_rate)
        return (
            transportation_cost(box_ids_F_P, box_ids_P_D, box_ids_D_R)
            + customer_illness_cost(box_ids_R_C, contamination_mask4)
            + testing_cost(tests_F, tests_P, tests_D, tests_R)
            + recall_cost(contamination_mask4)
        )

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


class Run(object):
    def __init__(self,parameters):
        self.parameters = parameters
    
    def initiate_farms(self):
        farms = []
    
    def initiate_supply_chain(self):
        self.supply = Supply(self.parameters)     

class Farms(object):
    def __init__(self,parameters):
        self.parameters = parameters
        self.farms = []
        self.farm_ids = []
        self.total_boxes = parameters.farm_population * parameters.plot_per_farm * parameters.box_per_plot
        self.box_ids = self.generate_box_ids_numpy(parameters.farm_population, parameters.plot_per_farm, parameters.box_per_plot)

    # New method to find the optimal testing rates for each stage
    def find_optimal_testing_rate(self, parameters):
        min_cost = float("inf")
        optimal_test_rates = None
        for f_test_rate in parameters.f_test_range:
            self.parameters.f_test_rate = f_test_rate
            for p_test_rate in parameters.p_test_range:
                self.parameters.p_test_rate = p_test_rate
                for d_test_rate in parameters.d_test_range:
                    self.parameters.d_test_rate = d_test_rate
                    for r_test_rate in parameters.r_test_range:
                        self.parameters.r_test_rate = r_test_rate
                        total_cost = self.run_test(parameters)
                        if total_cost < min_cost:
                            min_cost = total_cost
                            optimal_test_rates = (f_test_rate, p_test_rate, d_test_rate, r_test_rate)
        self.parameters.f_test_rate, self.parameters.p_test_rate, self.parameters.d_test_rate, self.parameters.r_test_rate = optimal_test_rates

    # Rest of the code remains the same as before

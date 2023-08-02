import numpy as np 
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

import parameters 
import gc
gc.enable()
 
class Farm:
    def __init__(self, id, technology_level, initial_contamination_rate, region_mean_contamination_rate):
        self.id = id
        self.technology_level = technology_level
        self.contamination_rate = initial_contamination_rate
        self.region_mean_contamination_rate = region_mean_contamination_rate
        self.cost_effort = 0
        self.cost_technology = 0
        self.penalties = [0, 0, 0, 0, 0]
        self.testing_regimes = [0, 0, 0, 0]
        
    def update_contamination_rate(self, c, k, t):
        # calculate new contamination rate based on Eq. (3)
        c_j_t = c[self.id][t]
        k_j_t = k[self.id][t]
        alpha_j_t = np.exp(-c_j_t * k_j_t)
        self.contamination_rate = alpha_j_t * self.region_mean_contamination_rate
        
    def calculate_cost(self, f, beta, P, c_e, c_k):
        # calculate total cost function based on Eq. (4)
        f1, f2, f3, f4, f5 = f
        beta1, beta2, beta3, beta4 = beta
        numerator = f1 * beta1 + f2 * (1 - beta1) * beta2 + f3 * (1 - beta1) * (1 - beta2) * beta3 + \
                    f4 * (1 - beta1) * (1 - beta2) * (1 - beta3) * beta4 + f5 * (1 - beta1) * (1 - beta2) * \
                    (1 - beta3) * (1 - beta4) * P
        denominator = (c_e + c_k) * (1 - beta1) * (1 - beta2) * (1 - beta3) * (1 - beta4)
        self.cost_effort = c_e
        self.cost_technology = c_k
        self.penalties = [f1, f2, f3, f4, f5]
        self.testing_regimes = [beta1, beta2, beta3, beta4]
        
    
        return numerator / denominator
        
    def update_risk_control_behavior(self, f, beta, P, c_e, c_k, c, k, t, neighbors):
        # update risk control behavior based on Eq. (5) and the behavior of neighboring farms
        cost = self.calculate_cost(f, beta, P, c_e, c_k)
        for neighbor in neighbors:
            cost += neighbor.calculate_cost(f, beta, P, c_e, c_k)
        self.cost_effort = c_e
        self.cost_technology = c_k
        self.penalties = [f1, f2, f3, f4, f5]
        self.testing_regimes = [beta1, beta2, beta3, beta4]
        self.contamination_rate = self.region_mean_contamination_rate
        self.contamination_rate += np.random.normal(loc=0, scale=self.technology_level)
        self.contamination_rate += np.random.uniform(low=-1, high=1) * c[self.id][t]
        self.contamination_rate += np.sum([np.random.uniform(low=-1, high=1) * neighbor.contamination_rate for neighbor in neighbors])
        self.update_contamination_rate(c, k, t)
        self.cost = self.calculate_cost(f, beta, P, c_e, c_k)

class Supply(object):
    def __init__(self,parameters):
        self.parameters = parameters
        self.farms = Farms(self.parameters)
        self.packers = Packers(self.parameters)
        self.distributors = Distributors(self.parameters)
        self.retailers = Retailers(self.parameters)
        self.customers = Customers(self.parameters)
    
    def update_contaminate_rate(self):
        self.farms.update_contaminate_rate()
    
    def find_optimal_testing_rate(self):
        self.farms.find_optimal_testing_rate()
    
    def testing_with_optimal_rate(self):
        self.farms.testing_with_optimal_rate()
    
    def impose_penalties_costs(self):
        self.farms.impose_penalties_costs()
        
class Monthly(object):
    def __init__(self,parameters):
        self.parameters = parameters
        self.monthly = Monthly(self.parameters)
    s
    def farmers_update_contaminate_rate(self):
        self.supply.update_contaminate_rate()
    
    def find_optimal_testing_rate(self):
        self.supply.find_optimal_testing_rate()
    
    def testing_with_optimal_rate(self):
        self.supply.testing_with_optimal_rate()
    
    def impose_penalties_costs(self):
        self.supply.impose_penalties_costs()

# Class to store parameters and initial conditions
class Parameters(object):
    def __init__(self) -> None:
        # Parameters for Farmer Behavior
        [beta1,beta2,beta3,beta4] = np.around(np.random.uniform(low=0.01, high=0.2, size=(4,)),4)
        [f1,f2,f3,f4] = np.random.rand(4)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.beta4 = beta4
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.f4 = f4
        
        # Population
        self.farm_population = 1
        self.packer_population = 1
        self.distributor_population = 1
        self.retailer_population = 1
        self.customer_population = 5000
        
        # Size of plot, box, and farm
        self.plot_per_farm = 1856
        self.box_per_P = 31562
        self.box_per_D = 31562
        self.box_per_R = 31562

        self.box_per_plot = 17
        self.contamination_rate = 0.06

        self.d_test_rate = 0.0
        self.r_test_rate = 0.0
        
        # Initialising some lists
        self.box_ids_l=[]
        self.box_ids_C_l=[]
        self.dropped_boxes_F_l=[]
        self.dropped_boxes_P_l=[]
        self.dropped_boxes_D_l=[]
        self.dropped_boxes_R_l=[]
        self.dropped_boxes_C_l=[]
        self.detection_percent_l=[]
        
        self.iterations = 10

        # Test cost at each stage
        self.test_cost_F = 350
        self.test_cost_P = 50
        self.test_cost_D = 50
        self.test_cost_R = 50

        # Customer demand
        self.box_cap = 50
        self.number_plot = self.farm_population * self.plot_per_farm
        self.number_box = self.number_plot * self.box_per_plot
        self.customer_number = int(self.number_box * self.box_cap * (0.80 - self.contamination_rate))
        self.customer_demand = np.floor(np.abs(np.random.normal(1, 2, (self.customer_number, 1))))

        # Customer illness cost
        self.ill_rate = 0.04
        self.hospital_rate = 0.0164
        self.death_rate = 0.000041
        self.ill_compensation = 719
        self.hospital_compensation = 18438
        self.death_compensation = 1764112

        # Recall and Trace cost
        self.unit_recall_labor_cost = 10
        self.unit_trace_labor_cost = 10
        self.price_per_box = 100

        # Transportation cost
        self.unit_trans_cost = 0.007
        self.cost_indicator = 1
        self.F_P_distance = 5
        self.P_D_distance = 2983
        self.D_R_distance = 11

import config 
import numpy as np
from src.box import*
from src.costs import*

farm_population = config.farm_population
packer_population = config.packer_population
distributor_population = config.distributor_population
retailer_population = config.retailer_population
customer_population = config.customer_population
plot_per_farm = config.plot_per_farm
box_per_plot = config.box_per_plot
contamination_rate = config.contamination_rate
d_test_rate = config.d_test_rate
r_test_rate = config.r_test_rate

# All boxes in same plot in which tested contaminated box is found are dropped
def run_stage(box_ids, contamination_mask, test_rate,box_per_plot):
    tested_boxes_mask = np.random.rand(box_ids.shape[0]) < test_rate
    tested_contaminated_boxes = box_ids[np.logical_and(contamination_mask, tested_boxes_mask)]
    if tested_contaminated_boxes.size > 0:
        plot_farm_ids_tested = tested_contaminated_boxes - tested_contaminated_boxes % 100
        #plot_farm_ids_all = box_ids - box_ids % 100
        tested_contaminated_boxes_all=np.repeat(np.unique(plot_farm_ids_tested),box_per_plot)+ np.tile(np.arange(1, box_per_plot + 1),np.unique(plot_farm_ids_tested).shape[0])
        #plot_boxes = np.array([generate_box_id(farm_idxs[i], plot_idxs[i], box_idx) for i in range(tested_contaminated_boxes.size) for box_idx in range(1, box_per_plot+1)])
        #remaining_boxes = box_ids[~np.isin(box_ids, dropped_boxes)]
        #tested_contaminated_boxes= np.array(list(dropped_boxes))
        dropped_boxes = set(np.unique(tested_contaminated_boxes_all))
        mask = np.isin(box_ids, tested_contaminated_boxes_all)
        contamination_mask = contamination_mask[~mask]
        box_ids_n = box_ids[~mask]
        #dropped_boxes = set(tested_contaminated_boxes)
    else:
        dropped_boxes = set()
        box_ids_n = box_ids
    remaining_boxes = box_ids_n

    return remaining_boxes, contamination_mask, dropped_boxes, sum(tested_boxes_mask)

# Run testing and return the number of boxes dropped at each stage
def run_test(p_test_rate, f_test_rate):
    total_boxes = farm_population * plot_per_farm * box_per_plot
    box_ids = generate_box_ids_numpy(farm_population, plot_per_farm, box_per_plot)
    contamination_mask = generate_contamination_mask(total_boxes, contamination_rate)
    box_ids_F_P, contamination_mask1, dropped_boxes_F,tests_F = run_stage(box_ids, contamination_mask, f_test_rate,box_per_plot)
    box_ids_P_D, contamination_mask2, dropped_boxes_P,tests_P = run_stage(box_ids_F_P, contamination_mask1, p_test_rate,box_per_plot)
    box_ids_D_R, contamination_mask3, dropped_boxes_D,tests_D = run_stage(box_ids_P_D, contamination_mask2, d_test_rate,box_per_plot)
    box_ids_R_C, contamination_mask4, dropped_boxes_R,tests_R = run_stage(box_ids_D_R, contamination_mask3, r_test_rate,box_per_plot)
    return (
        transportation_cost(box_ids_F_P, box_ids_P_D, box_ids_D_R)
        + customer_illness_cost(box_ids_R_C, contamination_mask4)
        + testing_cost(tests_F, tests_P, tests_D, tests_R)
        + recall_cost(contamination_mask4)
    )
    #return box_ids, box_ids_F_P, box_ids_P_D, box_ids_D_R, box_ids_R_C, contamination_mask, dropped_boxes_F, dropped_boxes_P, dropped_boxes_D, dropped_boxes_R



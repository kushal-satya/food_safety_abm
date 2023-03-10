
################################
####### COST FUNCTION #############
################################


# Farm side 
# Cost for farmer is dependednt on the effort and on technology level by the relation of cost = effort * technology_level
f_test_number = np.zeros(len(farm_id))
for i in range(len(farm_id)):
    f_test_number[i] = sum(contaminate_p[farm_box[i, 0] - 1 : farm_box[i, 1], iteration])

f_test_cost = f_test_number * f_test_effort * f_tech_level
    

test_number = sum(f_test_number) + sum(p_test_number) + sum(d_test_number) + sum(r_test_number)

unfound_contaminate = history_box_id_r[(contaminate_drop[:, iteration] == 0) & (contaminate_b[:, iteration] == 1)]
unfound_farmid = unfound_contaminate // 10000000
unfound_plotid = (unfound_contaminate - unfound_farmid * 10000000) // 1000

number_unfound = len(unfound_contaminate)

cur_customer = 1
contaminate_customer = []
for i in range(len(history_box_id_cus)):
    cur_farmid = history_box_id_cus[i, iteration] // 100000000000000
    cur_plotid = (history_box_id_cus[i, iteration] - cur_farmid * 100000000000000) // 100000000000
    if (cur_farmid in unfound_farmid) and (cur_plotid in unfound_plotid):
        contaminate_customer.append(history_box_id_cus[i, iteration])
        cur_customer += 1

# Customer side
customer_number = len(contaminate_customer)
ill_number = np.random.rand(customer_number)
hospital_number = np.random.rand(customer_number)
death_number = np.random.rand(customer_number)

death_number = death_number < death_rate
death_case_number = sum(death_number)
hospital_number

def safety_testing_cost()
    # Farm side
    farm_cost = 0
    for i in range(len(farm_id)):
        farm_cost += f_test_number[i] * f_test_cost[i]

    # Plot side
    plot_cost = 0
    for i in range(len(plot_id)):
        plot_cost += p_test_number[i] * p_test_cost[i]

    # Drop side
    drop_cost = 0
    for i in range(len(drop_id)):
        drop_cost += d_test_number[i] * d_test_cost[i]

    # Road side
    road_cost = 0
    for i in range(len(road_id)):
        road_cost += r_test_number[i] * r_test_cost[i]

    return farm_cost + plot_cost + drop_cost + road_cost

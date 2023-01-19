
################################
####### COST FUNCTION #############
################################

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

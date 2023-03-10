            ################################
            ####### DISTRIBUTOR TEST #############
            ################################

            d_test_number = np.random.rand(number_box, 1)
            d_test_number = d_test_number < d_test_rate

            d_test_history_box_id_cusult[:, iteration] = (d_test_number == 1 & contaminate_b[:, iteration] == 1 & contaminate_drop[:, iteration] == 0)
            d_test_location = np.where(d_test_number == 1 & contaminate_b[:, iteration] == 1 & contaminate_drop[:, iteration] == 0)

            d_test_number = (d_test_number == 1 & contaminate_drop[:, iteration] == 0)
            # How many boxes are tested in this process
            d_test_location_size = d_test_location.shape[0]

            # Drop contaminate box in same plot
            drop_plot_d = (d_test_location - 1) // box_per_plot + 1

            for i in range(number_box):
                for j in range(d_test_location_size):
                    if (i - 1) // box_per_plot == (d_test_location[j] - 1) // box_per_plot:
                        contaminate_drop[i, iteration] = 1

            # How many boxes will be delivered from distributor to retailor
            d_to_r_box = (contaminate_drop[:, iteration] == 0)

            # Add retailer ID
            retailer_id = []
            ind_d_to_r_dis = []
            cur_r_amount = 0
            cur_r_id = 1

            for i in range(history_box_id_d.shape[0]):
                if d_to_r_box[i] == 1:
                    while (cur_r_amount >= box_per_retailer[cur_r_id]):
                        cur_r_id += 1
                        cur_r_amount = 0
                    cur_r_amount += 1
                    history_box_id_r[i, iteration] = history_box_id_d[i, iteration] * 100 + cur_r_id
                else:
                    history_box_id_r[i, iteration] = history_box_id_d[i, iteration] * 100
                # Transportation cost
                retailer_id.append(history_box_id_r[i, iteration] % 100)
                if retailer_id[i] > 0:
                    ind_d_to_r_dis.append(d_r_distance[distributor_id[i], retailer_id[i]])
                else:
                    ind_d_to_r_dis.append(0)
            d_r_trans_cost[iteration] = unit_trans_cost * sum(ind_d_to_r_dis)

            ################################
            ####### PACKER TEST #############
            ################################
            
            p_test_number = np.random.rand(number_box, 1)
            p_test_number = p_test_number < p_test_rate

            p_test_history_box_id_cusult[:, iteration] = np.logical_and((p_test_number.flatten() == 1), (contaminate_b[:, iteration].flatten() == 1))

            p_test_location = np.argwhere((p_test_number == 1) & (contaminate_b[:, iteration] == 1) & (contaminate_drop[:, iteration] == 0))

            #p_test_history_box_id_cusult[:, iteration] = ((p_test_number == 1) & (contaminate_b == 1)).flatten()

            #p_test_location = np.argwhere(p_test_number == 1 & contaminate_b[:, iteration] == 1 & contaminate_drop[:, iteration] == 0)

            p_test_number = np.logical_and(p_test_number, contaminate_drop[:, iteration] == 0)


            # how many boxes are tested in this process
            p_test_location_size = p_test_location.shape[0]

            # drop contaminate box in same plot
            drop_plot_p = np.floor((p_test_location - 1) / box_per_plot) + 1
            contaminate_drop[(np.floor((np.arange(number_box) - 1) / box_per_plot) + 1) in drop_plot_p, iteration] = 1
            
            ''' 
            drop_plot_p = np.floor(p_test_location / box_per_plot).flatten()

            for i in range(number_box):
                for j in range(p_test_location_size):
                    if np.isin(np.floor(i / box_per_plot), np.floor(p_test_location[j] / box_per_plot)):
                        contaminate_drop[i, iteration] = 1
            ''' 
            # How many boxes will be delivered from packer to distributor
            p_to_d_box = contaminate_drop[:, iteration] == 0
            history_box_id_d[:iteration] = history_box_id_f[:iteration]

            farm_id = []
            distributor_id = []
            ind_f_to_d_dis = []

            cur_d_amount = 0
            cur_d_id = 1

            for i in range(len(history_box_id_f)):
                if p_to_d_box[i] == 1:
                    if cur_d_amount >= cur_d_id:
                        cur_d_id = cur_d_id + 1
                        cur_d_amount = 0
                    cur_d_amount = cur_d_amount + 1
                    history_box_id_d[i][iteration] = history_box_id_f[i][iteration] * 10 + cur_d_id
                else:
                    history_box_id_d[i][iteration] = history_box_id_f[i][iteration] * 10
                farm_id.append(int(history_box_id_d[i][iteration] // 100000))
                distributor_id.append(history_box_id_d[i][iteration] % 10)
                
                '''if distributor_id[i] > 0:
                    ind_f_to_d_dis.append(f_d_distance[farm_id[i]][distributor_id[i]])
                else:
                    ind_f_to_d_dis.append(0)
                '''
            f_d_trans_cost.append(unit_trans_cost * sum(ind_f_to_d_dis))

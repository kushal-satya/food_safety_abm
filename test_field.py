            ################################
            ####### FIELD TEST #############
            ################################
            # generate random array for f_test_number with f_test_rate
            f_test_number = np.random.rand(number_plot, 1)
            f_test_number = f_test_number < f_test_rate
            # create f_test_history_box_id_cusult and f_test_location arrays
            f_test_history_box_id_cusult[:, iteration] = ((f_test_number == 1) & (contaminate_p == 1)).flatten()
            
            f_test_location = np.argwhere(np.logical_and(f_test_number == 1, contaminate_p == 1))

            # get size of f_test_location array
            f_test_location_size = f_test_location.shape[0]

            # calculate drop_plot_f array and update contaminate_drop array

            drop_plot_f = np.floor((f_test_location - 1) / box_per_plot) + 1
            contaminate_drop[(np.floor((np.arange(number_box) - 1) / box_per_plot) + 1) in drop_plot_f, iteration] = 1

            ''' drop_plot_f = np.floor((f_test_location - 1) / box_per_plot) + 1
            for i in range(number_box):
                for j in range(f_test_location_size):
                    if np.floor((i-1) / box_per_plot) + 1 == f_test_location[j]:
                        contaminate_drop[i, iteration] = 1
            '''
            # create f_to_p_box array
            f_to_p_box = contaminate_drop == 0
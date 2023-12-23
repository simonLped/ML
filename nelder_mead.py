import numpy as np
from HDC_library import evaluate_F_of_x
from HDC_testing_library import evaluate_G_of_x

class nm:
    NM_iter = 100
    alpha_simp = 0.5
    gamma_simp = 1.2
    rho_simp = 0.3
    sigma_simp = 0.45
    Simplex = None
    STD_EPS = 0.002 #Threshold for early-stopping on standard deviation of the Simplex

class cf:
    F_of_x = None
    Accs = None
    Sparsities = None
    lambda_1 = 1
    lambda_2 = 1
    biases_ = None
    N_train = None
    D_b = 4
    B_cnt = 4
    Nbr_of_trials = 20

class data:
    HDC_cont_all = None
    LABELS = None

def nelder_mead(nm, cf = None, data = None, Testing = False):

    x_array = []
    y_array = []
    z_array = []

    F_of_x = cf.F_of_x
    Accs = cf.Accs
    Sparsities = cf.Sparsities

    Simplex = nm.Simplex

    if Testing: Simplex_record = np.zeros((len(Simplex), len(Simplex[0]), nm.NM_iter))

    STD_ = []
    objective_ = []
    for iter_ in range(nm.NM_iter):
        
        STD_.append(np.std(F_of_x))
        # objective_.append(np.mean(F_of_x))
        if STD_[-1] < nm.STD_EPS and 50 < iter_:
        # if STD_[-1] < nm.STD_EPS:
            break #Early-stopping criteria
        
        #1) sort Accs, Sparsities, F_of_x, Simplex, add best objective to array "objective_"
        Simplex = np.array(Simplex)
        sorted_indices = F_of_x.argsort()
        F_of_x_sorted = F_of_x[sorted_indices]
        Accs_sorted = Accs[sorted_indices]
        Sparsities_sorted = Sparsities[sorted_indices]
        Simplex_sorted = Simplex[sorted_indices]
        
        # print(F_of_x_sorted[0])

        # print(f"sorted {F_of_x_sorted}")
        # print(f"{Simplex_sorted}")

        #2) average simplex x_0 
        
        average_point = np.mean(Simplex_sorted[:-1, :], axis=0)
        # print(f"average point = {average_point}")
        #3) Reflexion x_r
        
        worst_point = Simplex_sorted[-1, :]
        reflected_point = average_point + nm.alpha_simp*(average_point - worst_point)
        #Evaluate cost of reflected point x_r

        gamma = reflected_point[0] #Regularization hyperparameter
        alpha_sp = reflected_point[1] #Threshold of accumulators
        beta_ = reflected_point[2] #incrementation step of accumulators
        if Testing:
            local_avg, reflected_local_avgre, reflected_local_sparse = evaluate_G_of_x(gamma, alpha_sp, beta_)
        else:
            local_avg, reflected_local_avgre, reflected_local_sparse = evaluate_F_of_x(cf.Nbr_of_trials, data.HDC_cont_all, data.LABELS, beta_, cf.biases_, gamma, alpha_sp, cf.N_train, cf.D_b, cf.lambda_1, cf.lambda_2, cf.B_cnt)
        # local_avg, reflected_local_avgre, reflected_local_sparse = evaluate_F_of_x(cf.Nbr_of_trials, data.HDC_cont_all, data.LABELS, beta_, cf.biases_, gamma, alpha_sp, cf.N_train, cf.D_b, cf.lambda_1, cf.lambda_2, cf.B_cnt)
       
        F_reflected = (1 - np.mean(local_avg)) #Append cost F(x)  

        # if F_reflected <= F_of_x_sorted[-2] and F_reflected > F_of_x_sorted[0]:
        if F_reflected < F_of_x_sorted[-2] and F_reflected >= F_of_x_sorted[0]:
            F_of_x_sorted[-1] = F_reflected
            Simplex_sorted[-1,:] = reflected_point
            Accs_sorted[-1] = np.mean(reflected_local_avgre)
            Sparsities_sorted[-1] = np.mean(reflected_local_sparse)
            print(f"reflection iter_={iter_}")
            # print(f"reflected_point {reflected_point}")
            rest = False
        else:
            rest = True
            
        if rest == True:
            #4) Expansion x_e
            
            if F_reflected < F_of_x_sorted[0]:

                expanded_point = average_point + nm.gamma_simp*(reflected_point - average_point)

                gamma = expanded_point[0] #Regularization hyperparameter
                alpha_sp = expanded_point[1] #Threshold of accumulators
                beta_ = expanded_point[2] #incrementation step of accumulators

                if Testing:
                    local_avg, expanded_local_avgre, expanded_local_sparse = evaluate_G_of_x(gamma, alpha_sp, beta_)
                else:
                    local_avg, expanded_local_avgre, expanded_local_sparse = evaluate_F_of_x(cf.Nbr_of_trials, data.HDC_cont_all, data.LABELS, beta_, cf.biases_, gamma, alpha_sp, cf.N_train, cf.D_b, cf.lambda_1, cf.lambda_2, cf.B_cnt)

                F_expanded = (1 - np.mean(local_avg)) #Append cost F(x) 
                
                if F_expanded < F_reflected:
                    F_of_x_sorted[-1] = F_expanded
                    Simplex_sorted[-1,:] = expanded_point
                    Accs_sorted[-1] = np.mean(expanded_local_avgre)
                    Sparsities_sorted[-1] = np.mean(expanded_local_sparse)
                    print(f"expansion iter_={iter_}")
                    # print(f"expanded point {expanded_point}")
                else:
                    F_of_x_sorted[-1] = F_reflected
                    Simplex_sorted[-1,:] = reflected_point
                    Accs_sorted[-1] = np.mean(reflected_local_avgre)
                    Sparsities_sorted[-1] = np.mean(reflected_local_sparse)
                    # print(f"reflection(exp) iter_={iter_}")
                    # print(f"reflected point {reflected_point}")
        
            else:
                inside = True
                #4) Contraction x_c
                if F_reflected < F_of_x_sorted[-1]:
                    contracted_point = average_point + nm.rho_simp*(reflected_point - average_point)
                    inside = False
                # elif F_reflected >= F_of_x_sorted[-1]:
                else:
                    contracted_point = average_point + nm.rho_simp*(worst_point - average_point)
                    inside = True
                    
                #Evaluate cost of contracted point x_e
                gamma = contracted_point[0] #Regularization hyperparameter
                alpha_sp = contracted_point[1] #Threshold of accumulators
                beta_ = contracted_point[2] #incrementation step of accumulators
              
                if Testing:
                    local_avg, contracted_local_avgre, contracted_local_sparse = evaluate_G_of_x(gamma, alpha_sp, beta_)
                else:
                    local_avg, contracted_local_avgre, contracted_local_sparse  = evaluate_F_of_x(cf.Nbr_of_trials, data.HDC_cont_all, data.LABELS, beta_, cf.biases_, gamma, alpha_sp, cf.N_train, cf.D_b, cf.lambda_1, cf.lambda_2, cf.B_cnt)
                
                F_contracted = (1 - np.mean(local_avg))
                
                if ((not inside) and F_contracted < F_reflected) or (inside and F_contracted < F_of_x_sorted[-1]):
                    F_of_x_sorted[-1] = F_contracted
                    Simplex_sorted[-1,:] = contracted_point
                    Accs_sorted[-1] = np.mean(contracted_local_avgre)
                    Sparsities_sorted[-1] = np.mean(contracted_local_sparse)
                    print(f"contraction iter_={iter_}")
                    # print(f"contracted point {contracted_point}")
                else:
                    #4) Shrinking
                    best_point = Simplex_sorted[0]
                    print(f"shrinking iter_={iter_}")
                    for i, current_point in enumerate(Simplex_sorted[1:]):
                        index = i + 1 #because we want to skip i = 0 - best point
                        shrunk_point = best_point + nm.sigma_simp*(current_point - best_point)
                        Simplex_sorted[index] = shrunk_point
                    F_of_x_temp = []
                    Accs_temp = []
                    Sparsities_temp = []
                    for point in range(len(Simplex)):
                        simp_arr = Simplex_sorted[point] #Take Simplex from list
                        gamma = simp_arr[0] #Regularization hyperparameter
                        alpha_sp = simp_arr[1] #Threshold of accumulators
                        beta_ = simp_arr[2] #incrementation step of accumulators

                        if Testing:
                            local_avg, local_avgre, local_sparse = evaluate_G_of_x(gamma, alpha_sp, beta_)
                        else:
                            local_avg, local_avgre, local_sparse = evaluate_F_of_x(cf.Nbr_of_trials, data.HDC_cont_all, data.LABELS, beta_, cf.biases_, gamma, alpha_sp, cf.N_train, cf.D_b, cf.lambda_1, cf.lambda_2, cf.B_cnt)

                        F_of_x_temp.append(1 - np.mean(local_avg)) #Append cost F(x)  
                        Accs_temp.append(np.mean(local_avgre))
                        Sparsities_temp.append(np.mean(local_sparse))

                        F_of_x_sorted = np.array(F_of_x_temp)
                        Accs_sorted = np.array(Accs_temp)
                        Sparsities_sorted = np.array(Sparsities_temp)
        objective_.append(F_of_x_sorted[0])
        Simplex = Simplex_sorted
        F_of_x = F_of_x_sorted
        Accs = Accs_sorted
        Sparsities = Sparsities_sorted

        # best_index = np.argmin(F_of_x)
        # x_array.append(Simplex_sorted[best_index, 0])
        # y_array.append(Simplex_sorted[best_index, 1])
        # z_array.append(Simplex_sorted[best_index, 2])
        

        worst_index = np.argmax(F_of_x)
        x_array.append(Simplex_sorted[worst_index, 0])
        y_array.append(Simplex_sorted[worst_index, 1])
        z_array.append(Simplex_sorted[worst_index, 2])

        # average_point = np.zeros(3)
        # average_point[0] = np.mean(Simplex_sorted[:-2, 0])
        # average_point[1] = np.mean(Simplex_sorted[:-2, 1])
        # average_point[2] = np.mean(Simplex_sorted[:-2, 2])
        
        # x_array.append(average_point[0])
        # y_array.append(average_point[1])
        # z_array.append(average_point[2])
        if Testing:
            Simplex_record[:, :, iter_] = Simplex
            np.save("Simplex_record", Simplex_record)

    return  F_of_x, Accs, Sparsities, Simplex, objective_, STD_, x_array, y_array, z_array
import numpy as np
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
from sklearn import svm # Used just to compare model
from sklearn.metrics import accuracy_score # Used just to compare model

# Generates random binary matrices of -1 and +1
# when mode == 0, all elements are generated with the same statistics (probability of having +1 always = 0.5)
# when mode == 1, all the probability of having +1 scales with an input "key" i.e., when the inputs to the HDC encoded are coded
# on e.g., 8-bit, we have 256 possible keys
def lookup_generate(dim, n_keys = 256, mode = 1):
    table = np.zeros([n_keys, dim]) 
    if mode == 0:
        for i in range(n_keys):
            table[i,:] = random.choices([1,-1], weights=[0.5, 0.5], k=dim)
    else:
        for i in range(n_keys):
            table[i,:] = random.choices([1,-1], weights=[i/(n_keys-1), 1-i/(n_keys-1)], k=dim)
            table[i,:] = random.choices([1,-1], weights=[i/(n_keys-1), 1-i/(n_keys-1)], k=dim)
                
    return table.astype(np.int8)  
        

def encode_HDC_RFF(img, position_table, grayscale_table, dim):

    img_LUT = position_table[img]
    img_hv = np.zeros(dim, dtype=np.int16) # dim=100 long vector
    container = np.zeros((len(img_LUT), dim)) # 30x100

    for pixel in range(grayscale_table.shape[0]):
    # for pixel in range(position_table.shape[0]):
        for k in range(dim):
            container[pixel,k]=grayscale_table[pixel,k]*img_LUT[pixel,k]*(-1) # XOR elementwise
        
    img_hv = np.sum(container, axis = 0) #bundling without the cyclic step yet
    return img_hv


# Receives the HDC encoded test set "HDC_cont_test" and test labels "Y_test"
# Computes test accuracy w.r.t. the HDC prototypes (centroids) and the biases found at training time
# Done with respect to -1 and 1 in the Labels, not 1 and 0
def compute_accuracy(HDC_cont_test, Y_test, centroid, bias):
   
    Acc = 0
    responses = np.zeros(Y_test.shape[0])
    for i in range(Y_test.shape[0]):
        current_HDC_vector = HDC_cont_test[i]
        inner_product = np.dot(centroid, current_HDC_vector) + bias
        responses[i] = 1 if inner_product >= 0 else -1
    Acc = np.sum(np.equal(Y_test, responses))
    
    return Acc/Y_test.shape[0]



# train_HDC_RFF_sk and compute_accuracy_sk is only to test with sklearn SVM training
def train_HDC_RFF_sk(Y_train, HDC_cont_train, gamma):

    clf = svm.SVC(kernel='linear').fit(HDC_cont_train, Y_train)
    # clf = svm.SVC(kernel='linear', C = gamma).fit(HDC_cont_train, Y_train)
    # clf = svm.LinearSVC().fit(HDC_cont_train, Y_train)
    return clf
    
def compute_accuracy_sk(HDC_cont_test, Y_test, clf):
    
    Y_predicted = clf.predict(HDC_cont_test)

    Acc = accuracy_score(Y_test, Y_predicted, normalize=True)

    return Acc



# Train the HDC circuit on the training set : (Y_train, HDC_cont_train)
# n_class: number of clases
# N_train: number of data points in training set
# gamma: LS-SVM regularization
# D_b: number of bit for HDC prototype quantization
def train_HDC_RFF(N_train, Y_train, HDC_cont_train, gamma, D_b):
    centroid = []
    centroid_q = []
    bias_q = []
    bias = []

    Beta = np.zeros((N_train + 1, N_train + 1))
    Beta[0, 1:] = Y_train.T
    Beta[1:, 0] = Y_train
    Omega = np.zeros((N_train, N_train))

    # for row  in range(0, N_train):
    #     for column in range(0, N_train):
    #         omega_ij = Y_train[row] * Y_train[column]  * (HDC_cont_train[row].T @ HDC_cont_train[column])
    #         Omega[row, column] = omega_ij
    Omega = np.outer(Y_train, Y_train) * (HDC_cont_train @ HDC_cont_train.T)

    Beta[1:, 1:] = Omega + np.identity(N_train)*(1/gamma)
    L = np.concatenate([[0], np.ones(N_train)])

    solution_vector = np.linalg.solve(Beta, L)
    b = solution_vector[0]
    alpha = solution_vector[1:]

    alpha_Y_intermediate = np.multiply(alpha, Y_train)
    final_HDC_centroid = np.matmul(alpha_Y_intermediate, HDC_cont_train)

    fact = 1/max(final_HDC_centroid)*(2**(D_b-1)-1)
    final_HDC_centroid_q = np.round(final_HDC_centroid * fact)
    
    #Amplification factor for the LS-SVM bias
    if np.max(np.abs(final_HDC_centroid)) == 0:
        # print("Kernel matrix badly conditionned! Ignoring...")
        centroid_q.append(np.ones(final_HDC_centroid_q.shape)) #trying to manage badly conditioned matrices, do not touch
        bias_q.append(10000)
        # centroid.append(np.ones(final_HDC_centroid_q.shape)) #trying to manage badly conditioned matrices, do not touch
        # bias.append(10000)
    else:
        centroid_q.append(final_HDC_centroid_q*1)
        bias_q.append(b*fact)
        # centroid.append(final_HDC_centroid_q*1)
        # bias.append(b*fact)
        
    centroid.append(final_HDC_centroid*1)
    bias.append(b)
        
    # return centroid, bias, centroid_q, bias_q
    return final_HDC_centroid, b, centroid_q, bias_q




#function that only does thresholding
def HDC_thresholding(HDC_cont_all, increment, biases_, B_cnt, alpha_sp):

    HDC_cont_all_cyclic_shifted = np.zeros(HDC_cont_all.shape)

    for row in range(len(HDC_cont_all)):
        HDC_cont_all_cyclic_shifted[row] = np.add(HDC_cont_all[row]*increment, biases_)%(2**B_cnt-1) -2**(B_cnt-1) #do a modulo increment on the whole matrix - cyclic accumulation
    HDC_thresholded = np.zeros(HDC_cont_all_cyclic_shifted.shape)

    HDC_thresholded[np.where(HDC_cont_all_cyclic_shifted > alpha_sp)] = 1
    HDC_thresholded[np.where(abs(HDC_cont_all_cyclic_shifted) <= alpha_sp)] = 0
    HDC_thresholded[np.where(HDC_cont_all_cyclic_shifted < -alpha_sp)] = -1

    return HDC_thresholded

# Evaluate the Nelder-Mead cost F(x) over "Nbr_of_trials" trials
# (HDC_cont_all, LABELS) is the complete dataset with labels
# increment is the output accumulator increment of the HDC encoder
# biases_ are the random starting value of the output accumulators
# gamma is the LS-SVM regularization hyper-parameter
# alpha_sp is the encoding threshold
# n_class is the number of classes, N_train is the number training points, D_b the HDC prototype quantization bit width
# lambda_1, lambda_2 define the balance between Accuracy and Sparsity: it returns lambda_1*Acc + lambda_2*Sparsity
 
# D_b: number of bit for HDC prototype quantization
# B_cnt: size of accumulator
def evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, increment, biases_, gamma, alpha_sp, N_train, D_b, lambda_1, lambda_2, B_cnt):
    local_avg = np.zeros(Nbr_of_trials)
    local_avgre = np.zeros(Nbr_of_trials)
    local_sparse = np.zeros(Nbr_of_trials)
    #Estimate F(x) over "Nbr_of_trials" trials
    for trial_ in range(Nbr_of_trials): 

        HDC_thresholded = HDC_thresholding(HDC_cont_all, increment, biases_, B_cnt, alpha_sp)

        HDC_cont_train = HDC_thresholded[:N_train,:] # Take training set
        Y_train = LABELS[:N_train]
        Y_train = Y_train.astype(int)

        # make testing data matrix
        HDC_cont_test = HDC_thresholded[N_train:,:]        
        Y_test = LABELS[N_train:]
        Y_test = Y_test.astype(int)


        centroids, biases, centroids_q, biases_q = train_HDC_RFF(N_train, Y_train, HDC_cont_train, gamma, D_b)
        # Acc = compute_accuracy(HDC_cont_test, Y_test, centroids_q, biases_q)
        # sparsity_HDC_centroid = np.array(centroids_q).flatten() 
        
        Acc = compute_accuracy(HDC_cont_test, Y_test, centroids_q, biases_q)
        sparsity_HDC_centroid = np.array(centroids_q).flatten() 
        nbr_zero = np.sum((sparsity_HDC_centroid == 0).astype(int))
        SPH = nbr_zero/(sparsity_HDC_centroid.shape[0])


        # -------------- Sk learn training and testing -------------
        # clf = train_HDC_RFF_sk(Y_train, HDC_cont_train, gamma)
        # centroids = clf.coef_
        # biases = clf.intercept_

        # Acc = compute_accuracy(HDC_cont_test, Y_test, centroids, biases)
        # # Acc = compute_accuracy(HDC_cont_train, Y_train, centroids, biases)
        # # Acc = compute_accuracy_sk(HDC_cont_test, Y_test, clf)
        # # Acc = compute_accuracy_sk(HDC_cont_train, Y_train, clf)
        # coefficients = clf.coef_
        # non_zero_coefficients = np.count_nonzero(coefficients)
        # total_coefficients = coefficients.size
        # SPH = (total_coefficients - non_zero_coefficients) / total_coefficients
        # ----------------------------------------------------------

        local_avg[trial_] = lambda_1 * Acc + lambda_2 * SPH #Cost F(x) is defined as 1 - this quantity
        local_avgre[trial_] = Acc
        local_sparse[trial_] = SPH
        
    return local_avg, local_avgre, local_sparse



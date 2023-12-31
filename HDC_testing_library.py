import numpy as np
import matplotlib.pyplot as plt
from HDC_library import lookup_generate, encode_HDC_RFF, HDC_thresholding, train_HDC_RFF, compute_accuracy
from scipy.stats import binomtest
from sklearn.utils import shuffle
import math
plt.close('all')

# test functions - maybe make library if there is many


# function tests if the statistical properties of the table
# generated by the lookup_generate function match theoretical expectations
# mode - 0/1
# nr_tests - number of tables to test
# level_of_significance - nthreshold (0,1) to reject the hypothesis that the generated table is fair
# H - height of the generated tables
# W - width of the generated tables

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#actually independent of z
def evaluate_G_of_x(x, y, z=0): 
    #two functions with minimum at (0,0,0)
    G1_of_x = x**2 + y**2
    G2_of_x = -math.exp(-(G1_of_x))
   
    G_of_x = 1-(G1_of_x + G2_of_x)
    return G_of_x, G1_of_x, G2_of_x

def test_lookup_generate(mode, nr_tests, maxval, H, W, level_of_significance = 0.05):
    test_ok = True
    current_table = np.zeros([maxval, W])
    summation_vector = np.zeros(H)
    accumulation_vector = np.zeros(H)
    
    if (mode == 0): #uniform
        for i in range(nr_tests):
            current_table = lookup_generate(W, maxval, mode = 0)
            current_table[current_table==-1] = 0
            summation_vector = np.sum(current_table, axis=1)
            plt.plot(summation_vector, label = f'Test# {i}')

            #the summation vector should follow a binomial distribution
            pvalue_vector = np.zeros(H)
            for j in range(H):
                experiment = binomtest(summation_vector[j], n=W, p=0.5)
                pvalue_vector[j] = experiment.pvalue
            p = np.sum(pvalue_vector)/H
            if (np.average(p) < level_of_significance):
                test_ok = False
        plt.axhline(W/2)
    else:   #grayscale
        for i in range(nr_tests):
            current_table = lookup_generate(W, maxval, mode = 1)
            current_table[current_table==-1] = 0
            summation_vector = np.sum(current_table, axis=1)
            plt.plot(summation_vector, label = f'Test# {i}')

            #the summation vector should follow a binomial distribution and corner cases need to be full -1 and 1
            pvalue_vector = np.zeros(H)
            for j in range(H):
                experiment = binomtest(summation_vector[j], n=W, p=j/H)
                pvalue_vector[j] = experiment.pvalue
            p = np.sum(pvalue_vector)/H
            test_ok = np.average(p) > level_of_significance and (current_table[0, :] == 0).all() and (current_table[maxval-1, :] == 1).all()
            plt.plot([0, H], [0, W])
    plt.title("Number of 1s in each row for all experiments")
    plt.xlim(0, H)
    plt.ylim(0, W)
    plt.ylabel("# of 1s")
    plt.xlabel("Row Index")
    plt.show()
    return(test_ok)


# this test can be deterministic since encode_HDC_RFF does not introduce randomness
def test_encode_HDC_RFF(D_HDC, imgsize, maxval):
    test_ok = True
    #generate sythetic W replacement - this is ok since encode_HDC_RFF does not introduce randomness
    W = np.concatenate((np.full((imgsize, int(D_HDC/2)), -1), np.ones((imgsize, int(D_HDC/2)))), axis=1)
    #generate real L - this is ok since we only test corner cases
    L = lookup_generate(D_HDC, maxval, mode = 1)
    #generate corner case images
    null_image = np.zeros(imgsize)
    print(W.shape)
    print(L.shape)
    print(null_image.shape)
    full_image = np.full(imgsize, maxval-1)
    #encode corener cases
    null_encoded = encode_HDC_RFF(null_image.astype(int), L, W, D_HDC)


    full_encoded = encode_HDC_RFF(full_image.astype(int), L, W, D_HDC)
    #since first and last rows of L are all 1 and -1 respectively the encoded vectors need to only follow the shape of the sythetic W
    test_ok_negative = (null_encoded[0:int(D_HDC/2)] == -imgsize).all() and (full_encoded[int(D_HDC/2): -1] == -imgsize).all() 
    test_ok_positive = (full_encoded[0:int(D_HDC/2)] ==  imgsize).all() and (null_encoded[int(D_HDC/2): -1] ==  imgsize).all() 
    test_ok = test_ok_negative and test_ok_positive
    #plotting
    plt.plot(null_encoded, label = "Encoded null vector")
    plt.plot(full_encoded, label = "Encoded ones vector")
    plt.title("Encoded corner cases")
    plt.legend()
    plt.xlim(0,D_HDC-1)
    plt.xlabel("Column intdex")
    plt.ylabel("Sum along column of the XOR matrix")
    plt.show()
    return test_ok


#testing thresholding
def test_HDC_thresholding(B_cnt):

    alpha_sp = 50 #arbitrary threshold value
    margin = 1

    ## under threshold - expected -1
    fill_value = -alpha_sp - margin + 2**(B_cnt-1)
    HDC_dummy =np.full((500,100), fill_value)
    increment = 1
    biases_ = np.zeros(100)
    HDC_dummy_thresholded = HDC_thresholding(HDC_dummy, increment, biases_, B_cnt, alpha_sp)
    assert (HDC_dummy_thresholded == -1 ).all(), f"{bcolors.FAIL}The HDC_thresholding() function has failed for the expected -1 value{bcolors.ENDC}"

    ## above threshold - expected +1
    fill_value = alpha_sp + margin + 2**(B_cnt-1)
    HDC_dummy =np.full((500,100), fill_value)
    increment = 1
    biases_ = np.zeros(100)
    B_cnt = 8
    HDC_dummy_thresholded = HDC_thresholding(HDC_dummy, increment, biases_, B_cnt, alpha_sp)
    assert (HDC_dummy_thresholded == 1).all(), f"{bcolors.FAIL}The HDC_thresholding() function has failed for the expected 1 value{bcolors.ENDC}"

    ## between threshold - expected 0
    #on the negative boundary
    fill_value = -alpha_sp + margin + 2**(B_cnt-1)
    HDC_dummy =np.full((500,100), fill_value)
    increment = 1
    biases_ = np.zeros(100)
    B_cnt = 8
    HDC_dummy_thresholded = HDC_thresholding(HDC_dummy, increment, biases_, B_cnt, alpha_sp)
    assert (HDC_dummy_thresholded == 0).all(), f"{bcolors.FAIL}The HDC_thresholding() function has failed for the expected 0 value on the negative boundary{bcolors.ENDC}"

    #on the positive boundary   
    fill_value = alpha_sp - margin + 2**(B_cnt-1)
    HDC_dummy =np.full((500,100), fill_value)
    increment = 1
    biases_ = np.zeros(100)
    B_cnt = 8
    HDC_dummy_thresholded = HDC_thresholding(HDC_dummy, increment, biases_, B_cnt, alpha_sp)
    assert (HDC_dummy_thresholded == 0).all(), f"{bcolors.FAIL}The HDC_thresholding() function has failed for the expected 0 value on the positive boundary{bcolors.ENDC}"

    #test bias effect - same as under threshold test but bias makes the value change the result
    fill_value = -alpha_sp - margin + 2**(B_cnt-1)
    HDC_dummy =np.full((500,100), fill_value)
    increment = 1
    biases_ = np.full(100, 2*margin)
    HDC_dummy_thresholded = HDC_thresholding(HDC_dummy, increment, biases_, B_cnt, alpha_sp)
    assert (HDC_dummy_thresholded == 0).all(), f"{bcolors.FAIL}The HDC_thresholding() function has failed for bias impact test{bcolors.ENDC}"

def test_train_HDC_RFF2(n_corruptions = 0, D_HDC = 100, B_cnt = 8, D_b = 8, realistic_mode = True, quantized = False):
    N_train = 2**(B_cnt-1)
    gamma = 0.0015
    n_class = 2
    #Initialize the sparsity-accuracy hyperparameter search

    phi_true = np.ones((N_train, D_HDC))
    Y_true = np.ones(N_train, dtype=int)
    # phi_false = -phi_true
    # Y_false = -Y_true

    phi = np.concatenate((phi_true, -phi_true))
    # Y = np.concatenate((Y_true, np.zeros(N_train)))
    Y = np.concatenate((Y_true, -Y_true))
    corruption_indices = np.zeros(n_corruptions, dtype=int)
    for i in range(N_train*2):
        corruption_indices = np.random.choice(int(D_HDC),size=n_corruptions, replace=False)
        for corruption in range (n_corruptions):
            if realistic_mode: phi[i, corruption_indices[corruption]] = -1 if phi[i, corruption_indices[corruption]] == np.random.choice([0,1]) else np.random.choice([-1,0])
            else: phi[i, corruption_indices[corruption]] = -1 if phi[i, corruption_indices[corruption]] == 1 else 1

    Y, phi = shuffle(Y, phi)
    centroids, biases, centroids_q, biases_q = train_HDC_RFF(N_train, Y[:N_train], phi[:N_train, :], gamma, D_b)

    if quantized: accuracy = compute_accuracy(phi[N_train:, :], Y[N_train:], centroids_q, biases_q)
    else: accuracy = compute_accuracy(phi[N_train:, :], Y[N_train:], centroids, biases)
    return accuracy == 1, accuracy


def test_train_HDC_RFF(N_train, D_HDC, D_b, quantization = False, Mode = 1):
    Half_of_array = round(N_train/2)
    gamma = 1
    if Mode:
        for_loop_iter = N_train
    else:
        for_loop_iter = 15*N_train

        


    # Create half array of -1 and half of 1
    Y_train = np.ones((N_train))
    Y_train[-Half_of_array:]   = -1

    # Create half array of -1 and half of 1
    HDC_cont_train = np.ones((N_train, D_HDC))
    HDC_cont_train[-Half_of_array:,:] = -1

    HDC_cont_train, Y_train = shuffle(HDC_cont_train, Y_train)

    accuracy = np.zeros(for_loop_iter)
    
    
    for i in range(for_loop_iter):

        # Every iteration missclasifies one more sample
        centroids, biases, centroids_q, biases_q = train_HDC_RFF(N_train, Y_train, HDC_cont_train, gamma, D_b)
        if quantization:
            accuracy[i] = compute_accuracy(HDC_cont_train, Y_train, centroids_q, biases_q)
        else:
            
            accuracy[i] = compute_accuracy(HDC_cont_train, Y_train, centroids, biases)

        if Mode:
            Y_train[i]  = Y_train[i]*(-1)
        else:
            random_integer = np.random.randint(low=0, high=256)
            Y_train[random_integer]  = Y_train[random_integer]*(-1)
            # Y_train[i]  = Y_train[i]*(-1)


    return accuracy
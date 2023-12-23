import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from HDC_library import lookup_generate, encode_HDC_RFF, evaluate_F_of_x
from nelder_mead import nelder_mead, nm, cf, data
from sklearn import svm
plt.close('all')
import time
import sys
np.set_printoptions(threshold=sys.maxsize)

start_time = time.time()


##################################   
#Replace the path "WISCONSIN/data.csv" with wathever path you have. Note, on Windows, you must put the "r" in r'C:etc..'
# dataset_path = 'WISCONSIN/data.csv' 
dataset_path = 'data.csv' 
##################################   
imgsize_vector = 30 #Each input vector has 30 features
n_class = 2
D_b = 4 #We target 4-bit HDC prototypes
B_cnt = 4
maxval = 256 #The input features will be mapped from 0 to 255 (8-bit)
D_HDC = 400 #HDC hypervector dimension
portion = 0.8 #We choose 60%-40% split between train and test sets
Nbr_of_trials = 20 #Test accuracy averaged over Nbr_of_trials runs
N_tradeof_points = 100 #Number of tradeoff points - use 100 
N_fine = int(N_tradeof_points*0.3) #Number of tradeoff points in the "fine-grain" region - use 30
#Initialize the sparsity-accuracy hyperparameter search
lambda_fine = np.linspace(-0.2, 0.2, N_tradeof_points-N_fine)
lambda_sp = np.concatenate((lambda_fine, np.linspace(-1, -0.2, N_fine//2), np.linspace(0.2, 1, N_fine//2)))
N_tradeof_points = lambda_sp.shape[0]



# Loading dataset
DATASET = np.loadtxt(dataset_path, dtype = object, delimiter = ',', skiprows = 1)
X = DATASET[:,2:].astype(float)
LABELS = DATASET[:,1]
LABELS[LABELS == 'M'] = -1
LABELS[LABELS == 'B'] = 1
LABELS = np.array(LABELS, dtype=np.int8)

# Scaling data
X = X / np.max(X, axis = 0)         # assuming max val is not 0
X = np.around(X*(maxval-1),0).astype(int)  # Each column is scaled from 0 to 255

X, LABELS = shuffle(X, LABELS)
imgsize_vector = X.shape[1]       # Number of features = 30
N_train = int(X.shape[0]*portion) # Number of training rows


"""
3) Generate HDC LUTs and bundle dataset
"""
grayscale_table = lookup_generate(D_HDC, imgsize_vector, mode = 0) #Weight-W_table for XOR-ing
position_table = lookup_generate(D_HDC, maxval, mode = 1) # LUT 0 to 255

HDC_cont_all = np.zeros((X.shape[0], D_HDC)) #Will contain all "bundled" HDC vectors
biases_ = np.round(np.random.rand(D_HDC) * (2**B_cnt)) #random biases

# Encoding data to phi(x)
for i in range(X.shape[0]):
    if i%100 == 0:
        print(str(i) + "/" + str(X.shape[0]))
    HDC_cont_all[i,:] = encode_HDC_RFF(X[i,:], position_table, grayscale_table, D_HDC)

 
print("HDC bundling finished...")

"""
4) Nelder-Mead circuit optimization and HDC training
"""
################################## 
#Nelder-Mead parameters
nm = nm()
cf = cf()
data = data()

nm.NM_iter = 300 #Maximum number of iterations
nm.STD_EPS = 0.002 #Threshold for early-stopping on standard deviation of the Simplex
#Contraction, expansion,... coefficients:
nm.alpha_simp = 1
nm.gamma_simp = 2
nm.rho_simp = 0.5
nm.sigma_simp = 0.5

################################## 

ACCS = np.zeros(N_tradeof_points)
SPARSES = np.zeros(N_tradeof_points)
load_simplex = False # Keep it to true in order to have somewhat predictive results

for optimalpoint in range(N_tradeof_points):
    print("Progress: " + str(optimalpoint+1) + "/" + str(N_tradeof_points))
    # F(x) = 1 - (lambda_1 * Accuracy + lambda_2 * Sparsity) : TO BE MINIMIZED by Nelder-Mead
    lambda_1 = 1 # Weight of Accuracy contribution in F(x)
    lambda_2 = lambda_sp[optimalpoint] # Weight of Sparsity contribution in F(x): varies!

    #Initialize the Simplex or either load a pre-defined one (we will use a pre-defined one in the lab for reproducibility)
    if load_simplex == False:
        Simplex = []
        N_p = 20
        for ii in range(N_p):
            alpha_sp = np.random.uniform(0, 1) * ((2**B_cnt) / 2)
            gam_exp = np.random.uniform(-5, -1)
            # print(gam_exp)
            beta_ = np.random.uniform(0, 2) * (2**B_cnt-1)/imgsize_vector
            gamma = 10**gam_exp
            simp_arr = np.array([gamma, alpha_sp, beta_])
            Simplex.append(simp_arr*1)  
    else:
        print("Loading simplex")
        Simplex = np.load("Simplex2.npz", allow_pickle = True)['data']
        Simplex = [row for row in Simplex]
    

    #Compute the cost F(x) associated to each point in the Initial Simplex
    F_of_x = []
    Accs = []
    Sparsities = []
    for init_simp in range(len(Simplex)):
        simp_arr = Simplex[init_simp] #Take Simplex from list
        gamma = simp_arr[0] #Regularization hyperparameter
        # alpha_sp = simp_arr[1] #Threshold of accumulators
        # beta_ = simp_arr[2] #incrementation step of accumulators
        ############### F(x) for Nelder-Mead with x = [gamma, alpha_sp, beta] ###################
        #The function "evaluate_F_of_x_2" performs:
        #    a) thresholding and encoding of bundled dataset into final HDC "ternary" vectors (-1, 0, +1)
        #    b) Training and testing the HDC system on "Nbr_of_trials" trials (with different random dataset splits)
        #    c) Returns lambda_1*Acc + lambda_2*Sparsity, Accuracy and Sparsity for each trials
        local_avg, local_avgre, local_sparse = evaluate_F_of_x(Nbr_of_trials, HDC_cont_all, LABELS, beta_, biases_, gamma, alpha_sp, N_train, D_b, lambda_1, lambda_2, B_cnt)
        F_of_x.append(1 - np.mean(local_avg)) #Append cost F(x)  
        Accs.append(np.mean(local_avgre))
        Sparsities.append(np.mean(local_sparse))
        ##################################

    nm.Simplex = np.array(Simplex)
    
    
    
    
    # For the details about the Nelder-Mead step, please refer to the course notes / reference, we are simply implementing that
    
    #prepare classes cf and data for nelder mead
    
    cf.F_of_x = np.array(F_of_x) 
    cf.Accs = np.array(Accs)
    cf.Sparsities = np.array(Sparsities)
    cf.lambda_1 = lambda_1
    cf.lambda_2 = lambda_2
    cf.biases_ = biases_
    cf.N_train = N_train
    cf.D_b = D_b
    cf.B_cnt = B_cnt
    cf.Nbr_of_trials = Nbr_of_trials

    data.HDC_cont_all = HDC_cont_all
    data.LABELS = LABELS

    F_of_x, Accs, Sparsities, Simplex, objective_nm, STD_nm, _, _, _ = nelder_mead(nm, cf, data)
    
    ################################## 
    #At the end of the Nelder-Mead search and training, save Accuracy and Sparsity of the best cost F(x) into the ACCS and SPARSES arrays
    idx = np.argsort(F_of_x)
    F_of_x = F_of_x[idx]
    Accs = Accs[idx]
    Sparsities = Sparsities[idx]
    Simplex = Simplex[idx, :]    

    ACCS[optimalpoint] = Accs[0]
    SPARSES[optimalpoint] = Sparsities[0]

    print(f"accuracy after point {optimalpoint} = {Accs[0]}")
    print(f"sparsity after point {optimalpoint} = {Sparsities[0]}")
    ################################## 

# Objective points and STD values for last lambda_ value
objective_ = objective_nm
STD_ = STD_nm

"""
Plot results (DO NOT TOUCH CODE)
Your code above should return:
    SPARSES: array with sparsity of each chosen lambda_
    ACCS: array of accuracy of each chosen lambda_
    objective_: array of the evolution of the Nelder-Mead objective of the last lambda_ under test
    STD_: array of the standard deviation of the simplex of the last lambda_ under test
    
"""

from datetime import datetime
current_datetime = datetime.now()
datetime_string = current_datetime.strftime('%Y-%m-%d_%H%M')
#save plotted parameters


np.save("Sparses_" + datetime_string, SPARSES, allow_pickle=True)
np.save("Accuracies_" + datetime_string, ACCS, allow_pickle=True)
np.save("Std_" + datetime_string, STD_, allow_pickle=True)
np.save("Objective_" + datetime_string, objective_, allow_pickle=True)
np.save("lambda_sp_" + datetime_string, lambda_sp, allow_pickle=True)

#Plot tradeoff curve between Accuracy and Sparsity
SPARSES_ = SPARSES[SPARSES > 0] 
ACCS_ = ACCS[SPARSES > 0]

print('SPARSES_')
print(SPARSES_)
print('ACCS_')
print(ACCS_)

end_time = time.time()
print(f"execution time = {end_time - start_time}")

plt.figure(1)
plt.plot(SPARSES_, ACCS_, 'x', markersize = 10)
plt.grid('on')
plt.xlabel("Sparsity")
plt.ylabel("Accuracy")

from sklearn.svm import SVR
y = np.array(ACCS_)
X = np.array(SPARSES_).reshape(-1, 1)
regr = SVR(C=1.0, epsilon = 0.005)
regr.fit(X, y)
X_pred = np.linspace(np.min(SPARSES_), np.max(SPARSES_), 100).reshape(-1, 1)
Y_pred = regr.predict(X_pred)
plt.plot(X_pred, Y_pred, '--')
plt.show()

#Plot the evolution of the Nelder-Mead objective and the standard deviation of the simplex for the last run
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(objective_, '.-')
plt.title("Objective")
plt.grid("on")
plt.subplot(2,1,2)
plt.plot(STD_, '.-') 
plt.title("Standard deviation") 
plt.grid("on")
plt.show()

plt.figure(3)
plt.plot(lambda_sp, ACCS)
plt.xlabel("Sparsity weight")
plt.ylabel("Accuracy")
plt.show()

plt.figure(4)
plt.plot(lambda_sp, SPARSES)
plt.xlabel("Sparsity weight")
plt.ylabel("Sparsity")
plt.show()



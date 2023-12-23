import numpy as np
import matplotlib.pyplot as plt
from HDC_testing_library import bcolors, test_train_HDC_RFF, test_HDC_thresholding, test_train_HDC_RFF2
import random
plt.close('all')


import sys
np.set_printoptions(threshold=sys.maxsize)


# imgsize_vector = 30 #Each input vector has 30 features
# n_class = 2
D_b = 4 #We target 4-bit HDC prototypes
B_cnt = 8
# maxval = 256 #The input features will be mapped from 0 to 255 (8-bit)
D_HDC = 100 #HDC hypervector dimension
# portion = 0.6 #We choose 60%-40% split between train and test sets
# Nbr_of_trials = 1 #Test accuracy averaged over Nbr_of_trials runs
# N_tradeof_points = 40 #Number of tradeoff points - use 100 
# N_fine = int(N_tradeof_points*0.4) #Number of tradeoff points in the "fine-grain" region - use 30
# #Initialize the sparsity-accuracy hyperparameter search
# lambda_fine = np.linspace(-0.2, 0.2, N_tradeof_points-N_fine)
# lambda_sp = np.concatenate((lambda_fine, np.linspace(-1, -0.2, N_fine//2), np.linspace(0.2, 1, N_fine//2)))
# N_tradeof_points = lambda_sp.shape[0]

print("Starting HDC_thresholding() test...")
test_HDC_thresholding(B_cnt)
print(f"{bcolors.OKGREEN}Test passed{bcolors.ENDC}")

print("Starting basic train_HDC_RFF() test...")
test_ok, _ = test_train_HDC_RFF2()
assert test_ok, f"{bcolors.FAIL}The test_train_HDC_RFF() or the compute_accuracy function has failed its basic test{bcolors.ENDC}"
print(f"{bcolors.OKGREEN}Test passed{bcolors.ENDC}")

# print("Starting systematic imperfect train_HDC_RFF() test with realistic corruptions...")
# accuracies = np.zeros(D_HDC)
# for i in range(D_HDC):
#     _, accuracies[i] = test_train_HDC_RFF2(n_corruptions=i)
# plt.plot(accuracies, label="Realistic corruptions")
# plt.grid()
# plt.xlim(0,D_HDC)
# plt.title("Accuracy vs. number of corruptions")
# plt.xlabel("Number of corruptions")
# plt.ylabel("Accuracy")
# plt.show()

print("Starting systematic imperfect train_HDC_RFF() test with artificial corruptions...")
#TODO: add automatic test result
accuracies = np.zeros(D_HDC)
for i in range(D_HDC):
    _, accuracies[i] = test_train_HDC_RFF2(n_corruptions=i, realistic_mode=False)
plt.plot(accuracies, label="Artificial corruptions")
plt.grid()
plt.xlim(0,D_HDC)
plt.title("Accuracy vs. number of corruptions")
plt.xlabel("Number of corruptions")
plt.ylabel("Accuracy")
plt.axhline(0.5, color='black')
plt.axvline(50, color='black')
# plt.legend()
plt.show()
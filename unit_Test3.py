import numpy as np
from HDC_testing_library import evaluate_G_of_x
import matplotlib.pyplot as plt
from nelder_mead import nelder_mead, nm, cf, data
from plotting import plot_Simplex


nm = nm()
nm.NM_iter = 50

#Compute the cost F(x) associated to each point in the Initial Simplex
F_of_x = []
Accs = []
Sparsities = []
Simplex = []
N_p = 4

cf = cf()

log2_runs = 3
runs = log2_runs**2
fig, axs = plt.subplots(log2_runs, log2_runs, figsize=(7, 7))

for run in range(runs):
    np.random.seed(run)
    F_of_x = []
    Accs = []
    Sparsities = []
    Simplex = []
    for j in range(N_p):
        x = np.random.uniform(5,50)
        y = np.random.uniform(5,50)
        z = 0
        simp_arr = np.array([x, y, z])
        Simplex.append(simp_arr*1)
        local_avg, local_avgre, local_sparse = evaluate_G_of_x(x, y, z)
        F_of_x.append(1 - np.mean(local_avg)) #Append cost F(x)  
        Accs.append(np.mean(local_avgre))
        Sparsities.append(np.mean(local_sparse))


    #Transform lists to numpy array:
    cf.F_of_x = np.array(F_of_x) 
    cf.Accs = np.array(Accs)
    cf.Sparsities = np.array(Sparsities)

    nm.Simplex = np.array(Simplex)
    print(nm.Simplex.shape)
    objective_ = [] #Will contain the objective value F(x) for each simplex as the Nelder-Mead search goes on
    STD_ = [] #Will contain the standard deviation of all F(x) as the Nelder-Mead search goes on

    _, _, _, _, _, _, x_array, y_array, z_array  = nelder_mead(nm, cf, Testing=True)


    axs[int(run/log2_runs), run%log2_runs].plot(x_array, label="x")
    axs[int(run/log2_runs), run%log2_runs].plot(y_array, label="y")
    axs[int(run/log2_runs), run%log2_runs].plot(z_array, label="z")
    axs[int(run/log2_runs), run%log2_runs].grid(which="both")
    axs[int(run/log2_runs), run%log2_runs].set_title(f"Starting point {run}", size=10)
    axs[int(run/log2_runs), run%log2_runs].autoscale(enable=True, tight=True)


axs[1,0].set_ylabel("Coordinate value")
axs[2,1].set_xlabel("NM iteration")
axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=5)
fig.tight_layout(pad=0.5)
# plt.autoscale(enable=True, tight=True)
plt.show()

plot_Simplex(merged=False)
plot_Simplex(merged=True)

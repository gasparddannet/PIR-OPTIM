import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True    

    #####################
    ###### DONNEES ######
    #####################

abs_15000 = [30, 40, 50]
abs_10000 = [20, 25, 30, 40, 50, 60, 70]
abs_8000 = [20, 25, 30, 40, 50, 60, 70]
abs_6000 = [20, 25, 30, 40, 50, 60, 70, 80]
abs_4000 = [20, 25, 30, 40, 50, 60, 70, 80]
abs_2000 = [25, 30, 40, 50, 60, 70, 80, 90, 100]
abs_1000 = [25, 30, 40, 50, 60, 70, 80, 90, 100]

pourcentage_erreur_15000 = [5.064, 1.671, 0.742]
pourcentage_erreur_10000 = [34.350, 12.230, 5.857, 1.925, 0.779, 0.478, 0.262]
pourcentage_erreur_8000 = [36.883, 12.613, 5.984, 2.155, 0.906, 0.464, 0.278]
pourcentage_erreur_6000 = [42.364, 12.863, 6.302, 2.150, 0.938, 0.543, 0.283, 0.217]
pourcentage_erreur_4000 = [48.411, 15.039, 7.528, 2.476, 1.120, 0.605, 0.311, 0.207]
pourcentage_erreur_2000 = [19.647, 9.806, 3.299, 1.276, 0.679, 0.386, 0.241, 0.179, 0.167]
pourcentage_erreur_1000 = [24.372, 11.683, 3.608, 1.652, 0.811, 0.482, 0.318, 0.231, 0.187]

temps_execution_15000 = [57.875, 70.465, 80.819]
temps_execution_10000 = [23.405, 24.952, 28.389, 34.786, 46.973, 58.472, 75.957]
temps_execution_8000 = [16.910, 18.695, 20.772, 26.313, 35.253, 46.939, 58.171]
temps_execution_6000 = [10.650, 11.969, 13.716, 18.115, 24.905, 32.634, 44.636, 57.590]
temps_execution_4000 = [5.332, 6.513, 7.302, 10.346, 14.992, 21.731, 28.561, 39.933]
temps_execution_2000 = [2.856, 3.263, 4.914, 7.350, 10.911, 15.572, 21.749, 28.728, 37.984]
temps_execution_1000 = [1.552, 1.825, 2.733, 3.987, 5.777, 8.067, 11.314, 14.598, 18.379]

param_15000 = "b.--"
param_10000 = "gP-"
param_8000 = "rx-"
param_6000 = "c<:"
param_4000 = "ms-"
param_2000 = "y^-"
param_1000 = "kd-."


# ------------------------------------------------------------------------------------------------------

abs_n = [1000, 2000, 4000, 6000, 8000, 10000]

temps_execution_en_fct_n_sr_25 = [1.552, 2.856, 6.513, 11.969, 18.695, 24.952]
temps_execution_en_fct_n_sr_30 = [1.825, 3.263, 7.302, 13.716, 20.772, 28.389]
temps_execution_en_fct_n_sr_40 = [2.723, 4.914, 10.346, 18.115, 26.313, 34.786]
temps_execution_en_fct_n_sr_50 = [3.987, 7.350, 14.992, 24.905, 35.523, 46.973]
temps_execution_en_fct_n_sr_60 = [5.777, 10.911, 21.731, 32.634, 46.939, 58.472]
temps_execution_en_fct_n_sr_70 = [8.067, 15.572, 28.561, 44.636, 58.171, 75.957]

pourcentage_erreur_en_fct_n_sr_25 = [24.372, 19.647, 15.039, 12.863, 12.613, 12.230]
pourcentage_erreur_en_fct_n_sr_30 = [11.683, 9.806, 7.528, 6.302, 5.984, 5.857]
pourcentage_erreur_en_fct_n_sr_40 = [3.608, 3.299, 2.476, 2.150, 2.155, 1.925]
pourcentage_erreur_en_fct_n_sr_50 = [1.652, 1.276, 1.120, 0.938, 0.906, 0.779]
pourcentage_erreur_en_fct_n_sr_60 = [0.811, 0.679, 0.605, 0.543, 0.464, 0.478]
pourcentage_erreur_en_fct_n_sr_70 = [0.482, 0.386, 0.311, 0.283, 0.278, 0.262]

param_25 = "b.--"
param_30 = "gP-"
param_40 = "rx-"
param_50 = "c<:"
param_60 = "ms-"
param_70 = "y^-"
param_1000 = "kd-."


# couleur avant
# ax2.plot(abs_15000, temps_execution_15000, label="n=15000", color="blue", marker = ".", linestyle = "--")
# ax2.plot(abs_10000, temps_execution_10000, label="n=10000", color = "darkorange", marker = "|", linestyle = "-")
# ax2.plot(abs_8000, temps_execution_8000, label="n=8000", color = "green", marker = "x", linestyle = "-")
# ax2.plot(abs_6000, temps_execution_6000, label="n=6000", color="red", marker = "<", linestyle = ":")
# ax2.plot(abs_4000, temps_execution_4000, label="n=4000", color = "purple", marker = "s")
# ax2.plot(abs_2000, temps_execution_2000, label="n=2000", color = "brown", marker = "^")
# ax2.plot(abs_1000, temps_execution_1000, label="n=1000", color = "magenta", marker = "d", linestyle = "-.")





    ###################
    ###### GRAPH ######
    ###################


# fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig = plt.figure(figsize=(15, 9))
spec = fig.add_gridspec(ncols=2, nrows=2)

ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1])
ax3 = fig.add_subplot(spec[1,0])
ax4 = fig.add_subplot(spec[1,1])


##################################################
############## Pourcentage d'erreur ##############
##################################################

## ax1.plot(abs_15000, pourcentage_erreur_15000, param_15000, label="n=15000")
# ax1.plot(abs_10000, pourcentage_erreur_10000, param_10000, label="n=10000")
# ax1.plot(abs_8000, pourcentage_erreur_8000, param_8000, label="n=8000")
# ax1.plot(abs_6000, pourcentage_erreur_6000, param_6000, label="n=6000")
# ax1.plot(abs_4000, pourcentage_erreur_4000, param_4000, label="n=4000")
# ax1.plot(abs_2000, pourcentage_erreur_2000, param_2000, label="n=2000")
# ax1.plot(abs_1000, pourcentage_erreur_1000, param_1000, label="n=1000")

# ax1.legend()
# ax1.set_xlabel("Search radius")
# ax1.set_ylabel("Pourcentage d'erreur")

# ------------------------------------------------------------------------------------------------------

ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_25, param_25, label="search radius = 25")
ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_30, param_30, label="sr=30")
ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_40, param_40, label="sr=40")
ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_50, param_50, label="sr=50")
ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_60, param_60, label="sr=60")
ax1.plot(abs_n, pourcentage_erreur_en_fct_n_sr_70, param_70, label="sr=70")

# ax1.tick_params(axis = 'both', labelsize = 15)
# ax1.legend(fontsize = 13)
ax1.legend()
ax1.set_xlabel("Number of samples")
ax1.set_ylabel("Error percentage")


##################################################
############ Temps d'exécution (en s) ############
##################################################

## ax2.plot(abs_15000, temps_execution_15000, param_15000, label="n=15000")
# ax2.plot(abs_10000, temps_execution_10000, param_10000, label="n=10000")
# ax2.plot(abs_8000, temps_execution_8000, param_8000, label="n=8000")
# ax2.plot(abs_6000, temps_execution_6000, param_6000, label="n=6000")
# ax2.plot(abs_4000, temps_execution_4000, param_4000, label="n=4000")
# ax2.plot(abs_2000, temps_execution_2000, param_2000, label="n=2000")
# ax2.plot(abs_1000, temps_execution_1000, param_1000, label="n=1000")

# ax2.legend()
# ax2.set_xlabel("Search radius")
# ax2.set_ylabel("Temps d'exécution (en s)")

# ------------------------------------------------------------------------------------------------------

ax2.plot(abs_n, temps_execution_en_fct_n_sr_25, param_25, label="sr=25")
ax2.plot(abs_n, temps_execution_en_fct_n_sr_30, param_30, label="sr=30")
ax2.plot(abs_n, temps_execution_en_fct_n_sr_40, param_40, label="sr=40")
ax2.plot(abs_n, temps_execution_en_fct_n_sr_50, param_50, label="sr=50")
ax2.plot(abs_n, temps_execution_en_fct_n_sr_60, param_60, label="sr=60")
ax2.plot(abs_n, temps_execution_en_fct_n_sr_70, param_70, label="sr=70")

ax2.legend()
ax2.set_xlabel("Number of samples")
ax2.set_ylabel("Execution time (in s)")


##################################################
## Rapport temps_execution / pourcentage_erreur ##
##################################################

# rapport_temps_execution_pourcentage_erreur_15000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_15000, pourcentage_erreur_15000)]
# rapport_temps_execution_pourcentage_erreur_10000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_10000, pourcentage_erreur_10000)]
# rapport_temps_execution_pourcentage_erreur_8000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_8000, pourcentage_erreur_8000)]
# rapport_temps_execution_pourcentage_erreur_6000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_6000, pourcentage_erreur_6000)]
# rapport_temps_execution_pourcentage_erreur_4000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_4000, pourcentage_erreur_4000)]
# rapport_temps_execution_pourcentage_erreur_2000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_2000, pourcentage_erreur_2000)]
# rapport_temps_execution_pourcentage_erreur_1000 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_1000, pourcentage_erreur_1000)]

# ## ax3.plot(abs_15000, rapport_temps_execution_pourcentage_erreur_15000, param_15000, label="n=15000")
# ax3.plot(abs_10000, rapport_temps_execution_pourcentage_erreur_10000, param_10000, label="n=10000")
# ax3.plot(abs_8000, rapport_temps_execution_pourcentage_erreur_8000, param_8000, label="n=8000")
# ax3.plot(abs_6000, rapport_temps_execution_pourcentage_erreur_6000, param_6000, label="n=6000")
# ax3.plot(abs_4000, rapport_temps_execution_pourcentage_erreur_4000, param_4000, label="n=4000")
# ax3.plot(abs_2000, rapport_temps_execution_pourcentage_erreur_2000, param_2000, label="n=2000")
# ax3.plot(abs_1000, rapport_temps_execution_pourcentage_erreur_1000, param_1000, label="n=1000")

# ax3.legend()
# ax3.set_xlabel("Search radius")
# ax3.set_ylabel("Rapport temps_execution/pourcentage_erreur")

# ------------------------------------------------------------------------------------------------------

rapport_temps_execution_pourcentage_erreur_sr_25 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_25, pourcentage_erreur_en_fct_n_sr_25)]
rapport_temps_execution_pourcentage_erreur_sr_30 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_30, pourcentage_erreur_en_fct_n_sr_30)]
rapport_temps_execution_pourcentage_erreur_sr_40 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_40, pourcentage_erreur_en_fct_n_sr_40)]
rapport_temps_execution_pourcentage_erreur_sr_50 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_50, pourcentage_erreur_en_fct_n_sr_50)]
rapport_temps_execution_pourcentage_erreur_sr_60 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_60, pourcentage_erreur_en_fct_n_sr_60)]
rapport_temps_execution_pourcentage_erreur_sr_70 = [temps_exec / pourcentage_erreur for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_70, pourcentage_erreur_en_fct_n_sr_70)]

ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_25, param_25, label="sr=25")
ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_30, param_30, label="sr=30")
ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_40, param_40, label="sr=40")
ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_50, param_50, label="sr=50")
ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_60, param_60, label="sr=60")
ax3.plot(abs_n, rapport_temps_execution_pourcentage_erreur_sr_70, param_70, label="sr=70")

ax3.legend()
ax3.set_xlabel("Number of samples")
ax3.set_ylabel(r"Ratio $ \frac{Execution time}{Error percentage} $")


##################################################
## Rapport pourcentage_erreur / temps_execution ##
##################################################

# rapport_pourcentage_erreur_temps_execution_15000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_15000, pourcentage_erreur_15000)]
# rapport_pourcentage_erreur_temps_execution_10000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_10000, pourcentage_erreur_10000)]
# rapport_pourcentage_erreur_temps_execution_8000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_8000, pourcentage_erreur_8000)]
# rapport_pourcentage_erreur_temps_execution_6000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_6000, pourcentage_erreur_6000)]
# rapport_pourcentage_erreur_temps_execution_4000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_4000, pourcentage_erreur_4000)]
# rapport_pourcentage_erreur_temps_execution_2000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_2000, pourcentage_erreur_2000)]
# rapport_pourcentage_erreur_temps_execution_1000 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_1000, pourcentage_erreur_1000)]

# ## ax4.plot(abs_15000, rapport_pourcentage_erreur_temps_execution_15000, param_15000, label="n=15000")
# ax4.plot(abs_10000, rapport_pourcentage_erreur_temps_execution_10000, param_10000, label="n=10000")
# ax4.plot(abs_8000, rapport_pourcentage_erreur_temps_execution_8000, param_8000, label="n=8000")
# ax4.plot(abs_6000, rapport_pourcentage_erreur_temps_execution_6000, param_6000, label="n=6000")
# ax4.plot(abs_4000, rapport_pourcentage_erreur_temps_execution_4000, param_4000, label="n=4000")
# ax4.plot(abs_2000, rapport_pourcentage_erreur_temps_execution_2000, param_2000, label="n=2000")
# ax4.plot(abs_1000, rapport_pourcentage_erreur_temps_execution_1000, param_1000, label="n=1000")

# ax4.legend()
# ax4.set_xlabel("Search radius")
# ax4.set_ylabel("Rapport pourcentage_erreur / temps_execution")

# ------------------------------------------------------------------------------------------------------

rapport_pourcentage_erreur_temps_execution_sr_25 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_25, pourcentage_erreur_en_fct_n_sr_25)]
rapport_pourcentage_erreur_temps_execution_sr_30 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_30, pourcentage_erreur_en_fct_n_sr_30)]
rapport_pourcentage_erreur_temps_execution_sr_40 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_40, pourcentage_erreur_en_fct_n_sr_40)]
rapport_pourcentage_erreur_temps_execution_sr_50 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_50, pourcentage_erreur_en_fct_n_sr_50)]
rapport_pourcentage_erreur_temps_execution_sr_60 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_60, pourcentage_erreur_en_fct_n_sr_60)]
rapport_pourcentage_erreur_temps_execution_sr_70 = [pourcentage_erreur / temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_70, pourcentage_erreur_en_fct_n_sr_70)]

ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_25, param_25, label="sr=25")
ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_30, param_30, label="sr=30")
ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_40, param_40, label="sr=40")
ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_50, param_50, label="sr=50")
ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_60, param_60, label="sr=60")
ax4.plot(abs_n, rapport_pourcentage_erreur_temps_execution_sr_70, param_70, label="sr=70")

ax4.legend()
ax4.set_xlabel("Sample numbers")
ax4.set_ylabel("Ration Error percentage / Ecexution time")

# ------------------------------------------------------------------------------------------------------

# mult_pourcentage_erreur_temps_execution_sr_25 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_25, pourcentage_erreur_en_fct_n_sr_25)]
# mult_pourcentage_erreur_temps_execution_sr_30 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_30, pourcentage_erreur_en_fct_n_sr_30)]
# mult_pourcentage_erreur_temps_execution_sr_40 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_40, pourcentage_erreur_en_fct_n_sr_40)]
# mult_pourcentage_erreur_temps_execution_sr_50 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_50, pourcentage_erreur_en_fct_n_sr_50)]
# mult_pourcentage_erreur_temps_execution_sr_60 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_60, pourcentage_erreur_en_fct_n_sr_60)]
# mult_pourcentage_erreur_temps_execution_sr_70 = [pourcentage_erreur * temps_exec for (temps_exec, pourcentage_erreur) in zip(temps_execution_en_fct_n_sr_70, pourcentage_erreur_en_fct_n_sr_70)]

# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_25, param_25, label="sr=25")
# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_30, param_30, label="sr=30")
# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_40, param_40, label="sr=40")
# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_50, param_50, label="sr=50")
# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_60, param_60, label="sr=60")
# ax4.plot(abs_n, mult_pourcentage_erreur_temps_execution_sr_70, param_70, label="sr=70")

# ax4.legend()
# ax4.set_xlabel("Sample numbers")
# ax4.set_ylabel(r"$\ Execution time * Error percentage $")



# plt.legend()
plt.show()
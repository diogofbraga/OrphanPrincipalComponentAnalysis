import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
kernel = "linear"


rmse_result_dfs = []
time_result_dfs = []

rmse_per_protein_df = []
time_per_protein_df = []


for draw in range(10):
    draw_folder = "../data/fixed_datasets/minLines_240_undersampling_True_numberDraws_10/" + str(draw)
    max_entry = pickle.load(open(draw_folder + "/max_entry.p", "rb"))
    minLines = pickle.load(open(draw_folder + "/minLines.p", "rb"))
    double_check = pickle.load(open(draw_folder + "/double_check.p", "rb"))
    maxProteins = pickle.load(open(draw_folder + "/maxProteins.p", "rb"))


    rmse_res_file = pd.read_csv(draw_folder+"/rmse_"+str(minLines)+"_Ligands_"+str(maxProteins)+"_MaxProteins_Kernel_"+kernel+"_DoubleCheck_"+str(double_check)+"_reg_1", sep=";", index_col=0, header=0)
    rmse_res = rmse_res_file[["NLCP", "OPCA", "SIMPLIFIED", "AVERAGED", "AVERAGED-3", "CLOSEST", "FARTHEST", "BESTPROTEIN", "SUPER5", "SUPER10", "SUPER30", "SUPER50", "SUPER80"]]

    rmse_per_protein_df.append(rmse_res)
    rmse_result_dfs.append(rmse_res.mean(axis=0))
    
    time_res_file = pd.read_csv(draw_folder+"/time_"+str(minLines)+"_Ligands_"+str(maxProteins)+"_MaxProteins_Kernel_"+kernel+"_DoubleCheck_"+str(double_check)+"_reg_1", sep=";", index_col=0, header=0)
    time_res = time_res_file[["NLCP", "OPCA", "SIMPLIFIED", "AVERAGED", "AVERAGED-3", "CLOSEST", "FARTHEST", "BESTPROTEIN", "SUPER5", "SUPER10", "SUPER30", "SUPER50", "SUPER80"]]

    time_per_protein_df.append(time_res)
    time_result_dfs.append(time_res.mean(axis=0))

rmse_per_protein_dict = {}
time_per_protein_dict = {}

# ------ RMSE ------ 
for protein in rmse_per_protein_df[0].index:
    rmse_per_protein_dict[protein] = {}
    for method in rmse_per_protein_df[0].columns.values:
        if method not in rmse_per_protein_dict:
            rmse_per_protein_dict[protein][method] = []
        for draw in range(10):
            rmse_per_protein_dict[protein][method].append(rmse_per_protein_df[draw].loc[protein][method])

# ------ TIME ------ 
for protein in time_per_protein_df[0].index:
    time_per_protein_dict[protein] = {}
    for method in time_per_protein_df[0].columns.values:
        if method not in time_per_protein_dict:
            time_per_protein_dict[protein][method] = []
        for draw in range(10):
            time_per_protein_dict[protein][method].append(time_per_protein_df[draw].loc[protein][method])
            

# ------ RMSE ------ 
for protein in rmse_per_protein_dict.keys():
    protein_dict = rmse_per_protein_dict[protein]
    protein_list = []
    for method in protein_dict.keys():
        protein_list.append(protein_dict[method])

    plt.figure()

    plt.title("Algorithm vs RMSE: "+protein)
    plt.boxplot(np.array(protein_list).transpose())
    ax = plt.gca()
    bars = ["NLCP", "OPCA", "Simplified CP", "Avg", "Avg-Clo-3", "Closest Protein", "Farthest Protein", "Best Protein", "Supervised-5%", "Supervised-10%", "Supervised-30%", "Supervised-50%", "Supervised-80%"]
    plt.xticks(np.arange(len(bars))+1,bars,rotation=45,fontsize=10, ha="right")
    plt.yticks(fontsize=10)
    ax.yaxis.grid(True,alpha=0.3,which="both")
    plt.xlabel('Algorithm', fontsize=10)
    plt.ylabel('RMSE', fontsize=10)
    plt.subplots_adjust(bottom=0.30)

    img_folder = "../data/fixed_datasets/minLines_240_undersampling_True_numberDraws_10/" + "img"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    plt.savefig(img_folder+"/average_rmse_protein"+protein+".png")
    plt.show()


# ------ TIME ------ 
for protein in time_per_protein_dict.keys():
    protein_dict = time_per_protein_dict[protein]
    protein_list = []
    for method in protein_dict.keys():
        protein_list.append(protein_dict[method])

    plt.figure()

    plt.title("Algorithm vs Execution Time: "+protein)
    plt.boxplot(np.array(protein_list).transpose())
    ax = plt.gca()
    bars = ["NLCP", "OPCA", "Simplified CP", "Avg", "Avg-Clo-3", "Closest Protein", "Farthest Protein", "Best Protein", "Supervised-5%", "Supervised-10%", "Supervised-30%", "Supervised-50%", "Supervised-80%"]
    plt.xticks(np.arange(len(bars))+1,bars,rotation=45,fontsize=10, ha="right")
    plt.yticks(fontsize=10)
    ax.yaxis.grid(True,alpha=0.3,which="both")
    plt.xlabel('Algorithm', fontsize=10)
    plt.ylabel('Execution Time (s)', fontsize=10)
    plt.subplots_adjust(bottom=0.30)

    img_folder = "../data/fixed_datasets/minLines_240_undersampling_True_numberDraws_10/" + "img"
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    plt.savefig(img_folder+"/average_time_protein"+protein+".png")
    plt.show()


rmse_res = np.array([r.to_numpy() for r in rmse_result_dfs])
time_res = np.array([r.to_numpy() for r in time_result_dfs])


bars = ["NLCP", "OPCA", "Simplified CP", "Avg", "Avg-Clo-3", "Closest Protein", "Farthest Protein", "Best Protein", "Supervised-5%", "Supervised-10%", "Supervised-30%", "Supervised-50%", "Supervised-80%"]

# ------ RMSE ------ 
from statistics import median
print("--- Median RMSE per algorithm ---")
print(np.median(rmse_res,axis=0))


plt.figure()
plt.title("Algorithm vs RMSE: all proteins")
plt.boxplot(rmse_res)
ax = plt.gca()
plt.xticks(np.arange(len(bars))+1,bars,rotation=45,fontsize=10, ha="right")
plt.yticks(fontsize=10)
ax.yaxis.grid(True,alpha=0.3,which="both")
plt.xlabel('Algorithm', fontsize=10)
plt.ylabel('RMSE', fontsize=10)
plt.subplots_adjust(bottom=0.30)
plt.savefig(img_folder + "/rmse_all_proteins")
plt.show()


# ------ TIME ------ 
from statistics import mean
mean_time_per_algorithm = time_res.mean(axis=0)
print("--- Mean time per algorithm ---")
print(mean_time_per_algorithm)

plt.figure()
plt.title("Algorithm vs Execution Time: all proteins")
plt.boxplot(time_res)
ax = plt.gca()
plt.xticks(np.arange(len(bars))+1,bars,rotation=45,fontsize=10, ha="right")
plt.yticks(fontsize=10)
ax.yaxis.grid(True,alpha=0.3,which="both")
plt.xlabel('Algorithm', fontsize=10)
plt.ylabel('Execution Time (s)', fontsize=10)
plt.subplots_adjust(bottom=0.30)
plt.savefig(img_folder + "/time_all_proteins")
plt.show()
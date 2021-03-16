import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import svr
import tanimotoKernel
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

# Root Mean Squared Error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

kernel = "linear"

for draw in range(10):

    print("------------------------ Draw", draw, "------------------------")

    draw_folder = "../data/fixed_datasets/minLines_240_undersampling_True_numberDraws_10/"+str(draw)

    protein_XY_dict = pickle.load(open(draw_folder+"/protein_XY_dict.p","rb"))
    compound_feature = pickle.load(open(draw_folder+"/compound_feature.p","rb"))
    order_per_protein = pickle.load(open(draw_folder+"/order_per_protein.p","rb"))


    # Calculate Gram matrix

    # Junction of the compounds of all proteins (rows) and, consequently, the features associated (columns)
    compound_matrix = []
    for i in range(len(compound_feature)):
        compound_matrix.append(compound_feature[i])
    compound_matrix = sparse.vstack(np.array(compound_matrix))

    print("Compound matrix shape:", compound_matrix.shape)

    # Gram_matrix = Similarities between all compounds at the level of features (number of features they have in common)
    gram_matrix = None
    if kernel == "linear":
        gram_matrix = np.dot(compound_matrix,compound_matrix.transpose())
        gram_matrix = gram_matrix.todense()
        gram_matrix = gram_matrix + np.ones(gram_matrix.shape) # So that there are no zeros
    elif kernel == "tanimoto":
        gram_matrix = tanimotoKernel.computeTanimotoMatrix(compound_matrix.todense())
        gram_matrix += np.ones(gram_matrix.shape)/15808

    print("Gram matrix", gram_matrix)
    print("Gram matrix shape:", gram_matrix.shape)

    # Store the best model per protein in a dictionary
    protein_models = {}

    # Create an SVR and optimize via gridsearch per protein
    for protein in protein_XY_dict:
        print("--------------- Protein:", protein, "---------------")
        kf = KFold(n_splits=10)
        X = protein_XY_dict[protein]["X"]
        y = protein_XY_dict[protein]["y"]
        param_grid = {"epsilon": [0.1, 0.01, 0.001], "C": [2 ** i for i in range(-5, 6)]}
        param_tuples = ParameterGrid(param_grid)

        # Reduce to the similarities between compounds of the protein selected in the cycle for (first: the rows, second: the columns)
        protein_gram = gram_matrix[order_per_protein[protein],:] 
        protein_gram = protein_gram[:,order_per_protein[protein]]
        print("Protein gram matrix shape:", protein_gram.shape)

        best_rmse = np.inf
        print("--------- Training ---------")
        for parameter in param_tuples:
            print("------ Parameter:", parameter, "------")
            p_rmse = []
            for train_index, test_index in kf.split(X):

                # Reduce to the similarities only for the train_index
                relevant_gram = protein_gram[train_index,:]
                relevant_gram = relevant_gram[:,train_index]

                # Get the ids (id = global compound index of the particular protein)
                Xids = np.array(order_per_protein[protein])
                Xids = Xids[train_index]


                # SVR train
                svs, coefs = svr.svr_train(relevant_gram, Xids, y[train_index], parameter["C"], parameter["epsilon"])
                # svs -> list of Xids that were selected based on the coefs resulted from the SRV algorithm
                # coefs -> list of selected coefs from the SRV algorithm

                # Reduce to the similarities only for the test_index and the indices (what indices?) returned by the svs
                K_test_svs = gram_matrix[order_per_protein[protein],:]
                K_test_svs = K_test_svs[test_index,:]
                K_test_svs = K_test_svs[:,svs]

                #support_vectors = np.array([compound_feature[v] for v in svs])
                #comparable_coefs =  sum([coefs[i]*support_vectors[i] for i in range(len(support_vectors))]).todense()

                #print("Sklearn coefs are {}".format(sk_svr.coef_))
                #print("Our SVR coefs are {}".format(comparable_coefs))

                #print("Difference is {}".format(np.sum(np.abs(sk_svr.coef_-comparable_coefs))))

                #print("Average differeonce per coeff is {}".format(average_diff))
                #print("Average coeff value is {}".format(average_coef))
                #print("Procentual difference per entry is {}".format(average_diff/average_coef))

                # Prediction (affinities) based on the coefs resulted from the train
                y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(test_index)))
                # Append of the different folds
                p_rmse.append(rmse(y_pred, y[test_index]))

            # Get the best model
            average_rmse = np.average(p_rmse)
            if best_rmse > average_rmse:
                Xids = np.array(order_per_protein[protein])
                svs, coefs = svr.svr_train(protein_gram, Xids, y, parameter["C"], parameter["epsilon"])
                # Add the best model for the particular protein
                protein_models[protein] = (svs,coefs)
                best_rmse = average_rmse
                print("New best RMSE for protein {} is {}".format(protein,best_rmse))

        print("Best RMSE for protein {} is {}".format(protein,best_rmse))


    with open(draw_folder+"/compound_matrix_"+kernel+".p","wb") as f:
        pickle.dump(compound_matrix,f)

    with open(draw_folder+"/gram_matrix_"+kernel+".p","wb") as f:
        pickle.dump(gram_matrix, f)

    with open(draw_folder+"/protein_models_"+kernel+".p","wb") as f:
        pickle.dump(protein_models,f)



import numpy as np
import pandas as pd
import svr
import pickle
from scipy.stats import kendalltau
import random
from sklearn.model_selection import ParameterGrid, KFold
import time


nan_values = 0

for draw in range(10):

    kernel = "linear"

    draw_folder = "../data/fixed_datasets/minLines_240_undersampling_True_numberDraws_10/"+str(draw)

    protein_XY_dict = pickle.load(open(draw_folder+"/protein_XY_dict.p","rb"))
    compound_feature = pickle.load(open(draw_folder+"/compound_feature.p","rb"))
    order_per_protein = pickle.load(open(draw_folder+"/order_per_protein.p","rb"))
    gram_matrix = pickle.load(open(draw_folder+"/gram_matrix_"+kernel+".p","rb"))
    protein_models = pickle.load(open(draw_folder+"/protein_models_"+kernel+".p","rb"))
    protein_sims = pickle.load(open(draw_folder+"/protein_sims.p","rb"))
    minLines = pickle.load(open(draw_folder+"/minLines.p","rb"))
    double_check = pickle.load(open(draw_folder+"/double_check.p","rb"))
    maxProteins = pickle.load(open(draw_folder+"/maxProteins.p","rb"))


    # Create a list of all protein names
    proteins = list(protein_XY_dict.keys())

    # Root Mean Squared Error
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())


    # ------------------------- Calculate OPCA, NLCP and other baselines -------------------------

    print("Draw {}".format(draw))

    # Write column names of all models
    rmse_file = open(draw_folder+"/rmse_"+str(minLines)+"_Ligands_"+str(maxProteins)+"_MaxProteins_Kernel_"+kernel+"_DoubleCheck_"+str(double_check)+"_reg_"+str(1),"w")
    rmse_file.write("Protein;NLCP;OPCA;SIMPLIFIED;AVERAGED;AVERAGED-3;CLOSEST;FARTHEST;BESTPROTEIN;SUPER5;SUPER10;SUPER30;SUPER50;SUPER80\n")

    kt_file = open(draw_folder+"/kt_"+str(minLines)+"_Ligands_"+str(maxProteins)+"_MaxProteins_Kernel_"+kernel+"_DoubleCheck_"+str(double_check)+"_reg_"+str(1),"w")
    kt_file.write("Protein;NLCP;OPCA;SIMPLIFIED;AVERAGED;AVERAGED-3;CLOSEST;FARTHEST;BESTPROTEIN;SUPER5;SUPER10;SUPER30;SUPER50;SUPER80\n")

    time_file = open(draw_folder+"/time_"+str(minLines)+"_Ligands_"+str(maxProteins)+"_MaxProteins_Kernel_"+kernel+"_DoubleCheck_"+str(double_check)+"_reg_"+str(1),"w")
    time_file.write("Protein;NLCP;OPCA;SIMPLIFIED;AVERAGED;AVERAGED-3;CLOSEST;FARTHEST;BESTPROTEIN;SUPER5;SUPER10;SUPER30;SUPER50;SUPER80\n")

    for orphan in proteins:

        print("------------------------------ Protein: {} ------------------------------".format(orphan))

        # The global indices of the bindings/compounds of the orphan protein
        orphan_indices = np.array(order_per_protein[orphan])

        # Within the global indices of the orphan, the (number of) indices which we want to use
        indices = list(range(minLines))

        # All proteins except for the orphan protein
        other_proteins = np.array([p for p in proteins if p != orphan])

        print("We have {} orphan indices and are using {} for prediction".format(len(orphan_indices),len(indices)))

        # Y-Values of the orphan protein
        y = protein_XY_dict[orphan]["y"]

        # SUPER50
        # Set of indices for validation and testing in the case of the supervised-50% model
        validation_index_50 = int(len(indices) * 0.5)
        validation_indices_50 = indices[0:validation_index_50]
        test_indices_50 = indices[validation_index_50:]

        # Y-Values of the orphan plus Y-Average benchmark
        y_validation_50 = protein_XY_dict[orphan]["y"][validation_indices_50]
        y_test_50 = protein_XY_dict[orphan]["y"][test_indices_50]

        # SUPER5
        # Set of indices for validation and testing in the case of the supervised-5% model
        validation_index_5 = int(len(indices) * 0.05)
        validation_indices_5 = indices[0:validation_index_5]
        test_indices_5 = indices[validation_index_5:]

        # Y-Values of the orphan plus Y-Average benchmark
        y_validation_5 = protein_XY_dict[orphan]["y"][validation_indices_5]
        y_test_5 = protein_XY_dict[orphan]["y"][test_indices_5]

        # SUPER10
        # Set of indices for validation and testing in the case of the supervised-10% model
        validation_index_10 = int(len(indices) * 0.1)
        validation_indices_10 = indices[0:validation_index_10]
        test_indices_10 = indices[validation_index_10:]

        # Y-Values of the orphan plus Y-Average benchmark
        y_validation_10 = protein_XY_dict[orphan]["y"][validation_indices_10]
        y_test_10 = protein_XY_dict[orphan]["y"][test_indices_10]

        # SUPER30
        # Set of indices for validation and testing in the case of the supervised-30% model
        validation_index_30 = int(len(indices) * 0.3)
        validation_indices_30 = indices[0:validation_index_30]
        test_indices_30 = indices[validation_index_30:]

        # Y-Values of the orphan plus Y-Average benchmark
        y_validation_30 = protein_XY_dict[orphan]["y"][validation_indices_30]
        y_test_30 = protein_XY_dict[orphan]["y"][test_indices_30]

        # SUPER80
        # Set of indices for validation and testing in the case of the supervised-80% model
        validation_index_80 = int(len(indices) * 0.8)
        validation_indices_80 = indices[0:validation_index_80]
        test_indices_80 = indices[validation_index_80:]

        # Y-Values of the orphan plus Y-Average benchmark
        y_validation_80 = protein_XY_dict[orphan]["y"][validation_indices_80]
        y_test_80 = protein_XY_dict[orphan]["y"][test_indices_80]

        print("-------------- Metrics comparison --------------")

        # Metrics comparison
        other_y_s = []
        for op in other_proteins:
            for y_value in protein_XY_dict[op]["y"]:
                other_y_s.append(y_value)
        y_other_average = np.average(other_y_s)
        y_other_std = np.std(other_y_s)

        print("Average y of orphan protein is: {}".format(np.average(y)))
        print("Average standard deviation of y of orphan protein is: {}".format(np.std(y)))
        print("Average y of other proteins is: {}".format(y_other_average))
        print("Average standard deviation of y of other proteins is: {}".format(y_other_std))

        # Y_other_average times length of the y, to compare the rmse and kt
        average_y = np.array([y_other_average]*len(y))
        average_y_rmse = rmse(np.array(y), average_y)
        average_y_kt = kendalltau(np.array(y), average_y)[0]
        print("RMSE between y and y average of the other proteins is {}".format(average_y_rmse))

        print("-------------- Global Protein Models Matrix --------------")

        # Building PI-Matrix for proteins - Preparation of the use of the models (indexes and coefficients) of other proteins besides the orphan
        pi_matrix = np.array([])
        pi_list = []
        for op in other_proteins:
            pi = np.zeros((gram_matrix.shape[0], 1))
            # protein_models[op][0] = list of support vector indices (order of the compounds in a protein combination (after Train_SVMs))
            # protein_models[op][1] = list of support vector coefficients
            # Obs: There are no indices in common
            # Assign each coefficient to the respective index (for each protein)
            pi[protein_models[op][0], 0] = protein_models[op][1]

            # Add each protein model to a global protein models list of matrices
            pi_list.append(pi)

        # Create a matrix from the global protein models (indexes of the combinations and respective coefficients) list (of matrices)
        pi_matrix = np.hstack(pi_list).reshape(pi.shape[0], len(pi_list))
        print("PI matrix shape:", pi_matrix.shape)

        factors = [] # To store the similarities between proteins (only in relation to the orphan)

        # For all other proteins, obtain their similarities to orphan protein
        for op in other_proteins:
            factors.append(protein_sims[orphan][op])

        factors = np.array(factors)/sum(factors)

        # Get the closest 3 proteins
        closest_3_positions = factors.argsort()[-3:][::-1]
        closest_3_factors = factors[closest_3_positions]
        closest_3_proteins = np.array(other_proteins)[closest_3_positions]

        # Building PI-Matrix for closest 3 for protein (do we need to calculate a seperate gram-matrix?)
        closest_3_pi_matrix = np.array([])
        pi_list = []
        for op in closest_3_proteins:
            pi = np.zeros((gram_matrix.shape[0], 1))
            pi[protein_models[op][0], 0] = protein_models[op][1]
            pi_list.append(pi)
        closest_3_pi_matrix = np.hstack(pi_list).reshape(pi.shape[0], len(pi_list))

        # Closest Protein
        closest_protein_index = np.argmax(factors)
        closest_protein = other_proteins[closest_protein_index]

        # Farthest Protein
        farthest_protein_index = np.argmin(factors)
        farthest_protein = other_proteins[farthest_protein_index]


        # ------------------------- Application of the models -------------------------

        print("-------------- Application of the models --------------")

        # -------- Model: Closest Protein -------- 
        start = time.time()
        p_closest = np.dot(gram_matrix, pi_matrix[:, closest_protein_index])
        p_closest = np.squeeze(np.asarray(p_closest))
        closest_est_predictions = p_closest[orphan_indices]
        closest_est_predictions = p_closest[indices]
        closest_est_predictions = closest_est_predictions.reshape(1,closest_est_predictions.shape[0])
        closest_rmse = rmse(np.array(y), np.array(closest_est_predictions))
        closest_kt = kendalltau(np.array(y),np.array(closest_est_predictions))[0]
        end = time.time()
        closest_time = end - start

        # -------- Model: Farthest Protein -------- 
        start = time.time()
        p_farthest = np.dot(gram_matrix, pi_matrix[:, farthest_protein_index])
        p_farthest = np.squeeze(np.asarray(p_farthest))
        farthest_est_predictions = p_farthest[orphan_indices]
        farthest_est_predictions = p_farthest[indices]
        farthest_est_predictions = farthest_est_predictions.reshape(1,farthest_est_predictions.shape[0])
        farthest_rmse = rmse(np.array(y), np.array(farthest_est_predictions))
        farthest_kt = kendalltau(np.array(y), np.array(farthest_est_predictions))[0]
        end = time.time()
        farthest_time = end - start

        # -------- Model: NLCP -------- 

        start = time.time()
        # Find best nu
        regularization_term = 1


        # CALCULATING NLCP
        # Pi.T * K * Pi
        G = np.dot(pi_matrix.T,np.dot(gram_matrix,pi_matrix))
        N = np.identity(G.shape[1])

        # nu * G
        #left_term = regularization_term *  np.diag(np.diag(G)) # Gecko did this. This is a matrix with only the diagonal of G, the rest is zero
        left_term = regularization_term * G

        # G * N * G
        middle_term = np.dot(G,np.dot(N,G))

        #
        A = left_term + middle_term

        I = np.eye(A.shape[0])

        lmbd = min(np.linalg.eig(A)[0]) # Mininum of: Compute the eigenvalues and right eigenvectors of a square array.


        # [nu * G + G*N*G]^-1
        left_term = np.linalg.inv(A+lmbd*I)

        # rho_0 = k_T * sqrt({G}_ii)
        # We use protein_sims because we want comparison only between the orphan and the rest
        rho_0 = np.array([protein_sims[orphan][other_proteins[i]] * np.sqrt(G[i,i]) for i in range(len(other_proteins))]).reshape(len(other_proteins),1)

        # beta_0 = [nu * G + G*N*G]^-1 * G +rho_0
        beta_0 = np.dot(left_term,np.dot(G,rho_0)) # np.dot(G,rho_0) is a junction of G (similarities obtained before) + the similarities of proteins at the beginning (protein_sims)
        beta_0 /= np.linalg.norm(beta_0,1) # Matrix or vector norm.

        # h_0(x) = {K * Pi_Matrix * beta_0}x
        h_0 = np.dot(gram_matrix,np.dot(pi_matrix,beta_0))
        #print("h_0 shape:", h_0.shape)


        nlcp_est_predictions = h_0[orphan_indices, :]
        nlcp_est_predictions = nlcp_est_predictions[indices, :]
        nlcp_est_predictions = nlcp_est_predictions.reshape(1, nlcp_est_predictions.shape[0])
        nlcp_rmse = rmse(np.array(y), np.array(nlcp_est_predictions))
        nlcp_kt = kendalltau(np.array(y), np.array(nlcp_est_predictions))[0]
        end = time.time()
        nlcp_time = end - start

        # -------- Model: OPCA -------- 
        start = time.time()

        # 1) ----- Creation of Gram matrix K of k (targets and hypothesis included) ----- 

        # Targets
        targets = np.append(other_proteins, orphan) # Orphan target is the last row and column
        gram_targets_matrix_opca = protein_sims[targets][:]
        gram_targets_matrix_opca = gram_targets_matrix_opca.loc[targets]

        # Hypothesis
        pi_hypothesis_matrix_opca = np.dot(pi_matrix.T, np.dot(gram_matrix, pi_matrix))

        # Upside
        upside_zeros = np.zeros((gram_targets_matrix_opca.shape[0], pi_hypothesis_matrix_opca.shape[1]))
        upside = np.hstack((gram_targets_matrix_opca, upside_zeros))

        # Downside
        downside_zeros = np.zeros((pi_hypothesis_matrix_opca.shape[0], gram_targets_matrix_opca.shape[1]))
        downside = np.hstack((downside_zeros, pi_hypothesis_matrix_opca))

        K_gram_matrix = np.vstack((upside, downside))
        #np.savetxt("KGram_matrix_"+str(orphan)+".csv", K_gram_matrix, delimiter=",")

        #print("KGram matrix shape:", K_gram_matrix.shape)

        # ----- Set C = {Protein: [(line of the target values, line of the hypothesis values), (target values, hypothesis values)], ... } ----- 

        C = {}
        number_of_targets = len(targets)
        i = 0
        for x in other_proteins:
            target = np.squeeze(np.asarray(K_gram_matrix[i]))
            target = target[:number_of_targets]
            hypothesis = np.squeeze(np.asarray(K_gram_matrix[number_of_targets+i]))
            hypothesis = hypothesis[number_of_targets:]
            C[x] = [(i, number_of_targets + i),(target, hypothesis)]
            i = i+1

        # 2) ----- Matrix HD ----- 

        D = K_gram_matrix.shape[0]

        I = np.identity(D)
        
        H = I - ((1/D) *  np.dot(np.ones((D,D)), np.ones((D,D)).T))

        # 3) ----- Matrix L ----- 

        L = np.zeros((K_gram_matrix.shape))

        for protein in C:
            target_line = C[protein][0][0]
            hypothesis_line = C[protein][0][1]
            el = np.zeros(D)
            el[target_line] = 1
            ell = np.zeros(D)
            ell[hypothesis_line] = 1
            L = L + np.outer((el - ell),(el - ell))

        # 4) ----- Optimal projection ----- 

        # Hyperparameter
        v = 1

        from scipy.optimize import minimize

        def objective(x):
            functionValue = (((np.trace(((((x.T).dot(K_gram_matrix)).dot(H)).dot(K_gram_matrix)).dot(x)) / D) - ((v * np.trace(((((x.T).dot(K_gram_matrix)).dot(L)).dot(K_gram_matrix)).dot(x))) / len(C))) - (100 * np.trace((((x.T).dot(K_gram_matrix)).dot(x) - I))))
            res = -1 * functionValue
            return res

        shape = K_gram_matrix.shape[0]

        def grad(x):
            res1 = ((1 / D) * (((K_gram_matrix).dot(H)).dot(K_gram_matrix)).dot(x))
            res2 = ((1 / D) * ((K_gram_matrix).dot((H).dot(K_gram_matrix)).T).dot(x))
            res3 = (((v / len(C)) * (((K_gram_matrix).dot(L)).dot(K_gram_matrix)).dot(x)) + ((v / len(C)) * ((K_gram_matrix).dot((L).dot(K_gram_matrix)).T).dot(x)))
            res4 = ((100 * (K_gram_matrix).dot(x)) + (100 * (K_gram_matrix.T).dot(x)))
            res = res1 + res2 - res3 - res4
            res = -1 * res
            res = np.copy(res)
            res = res.reshape(-1)
            return res

        x0 = np.random.rand(shape, shape).ravel()

        def grad_fun(x):
            res = grad(x.reshape(shape,shape))
            return res

        def obj_fun(x):
            res = objective(x.reshape(shape,shape))
            return res

        def con(x):
            return np.dot(x.T, np.dot(K_gram_matrix, x)) - I

        def con_fun(x):
            return con(x.reshape(shape,shape))
        
        cons = [{'type':'eq', 'fun': con_fun}]

        sol = minimize(obj_fun, x0, method='BFGS', jac = grad_fun, options={'maxiter': 1000})

        result = sol.x

        # To check for errors in the matrix created in the optimization 
        if np.isnan(np.min(result)):
            nan_values = nan_values + 1

        optimal_projection = result.reshape(shape,shape)
        #print("Optimal projection:", optimal_projection.shape)


        # 5) ----- Orphan hypothesis ----- 

        best_result = np.inf
        i = 0
        for protein in C:
            hypothesis = np.squeeze(np.asarray(K_gram_matrix[number_of_targets+i]))
            i = i+1
            orphan_target = np.squeeze(np.asarray(K_gram_matrix[number_of_targets-1]))
            result = np.power(np.linalg.norm(np.dot((hypothesis - orphan_target), optimal_projection)),2)
            if result < best_result:
                best_result = result
                best_hypothesis = hypothesis
                best_orphan_protein = protein
        
        #print("Best protein:", best_orphan_protein)

        # 6) ----- RMSE ----- 

        opca_best_protein_index = np.where(other_proteins == best_orphan_protein)

        opca_best = np.dot(gram_matrix, pi_matrix[:, opca_best_protein_index[0][0]])
        opca_best = np.squeeze(np.asarray(opca_best))
        opca_est_predictions = opca_best[orphan_indices]
        opca_est_predictions = opca_best[indices]
        opca_est_predictions = opca_est_predictions.reshape(1, opca_est_predictions.shape[0])
        opca_rmse = rmse(np.array(y), np.array(opca_est_predictions))
        opca_kt = kendalltau(np.array(y),np.array(opca_est_predictions))[0]
        end = time.time()
        opca_time = end - start

        # -------- Model: Simplified -------- 
        start = time.time()
        h_simplified = np.dot(gram_matrix,np.dot(pi_matrix,factors.reshape(factors.shape[0],1)))
        simplified_est_predictions = h_simplified[orphan_indices, :]
        simplified_est_predictions = simplified_est_predictions[indices, :]
        simplified_est_predictions = simplified_est_predictions.reshape(1, simplified_est_predictions.shape[0])
        simplified_rmse = rmse(np.array(y),np.array(simplified_est_predictions))
        simplified_kt = kendalltau(np.array(y),np.array(simplified_est_predictions))[0]
        end = time.time()
        simplified_time = end - start

        # -------- Model: Averaged -------- 
        start = time.time()
        uniform_factors = np.ones((factors.shape[0],1))/len(factors)
        h_averaged = np.dot(gram_matrix,np.dot(pi_matrix,uniform_factors))
        averaged_est_predictions = h_averaged[orphan_indices, :]
        averaged_est_predictions = averaged_est_predictions[indices, :]
        averaged_est_predictions = averaged_est_predictions.reshape(1, averaged_est_predictions.shape[0])
        averaged_rmse = rmse(np.array(y),np.array(averaged_est_predictions))
        averaged_kt = kendalltau(np.array(y),np.array(averaged_est_predictions))[0]
        end = time.time()
        averaged_time = end - start

        # -------- Model: Averaged-Closest-3 -------- 
        start = time.time()
        uniform_closest_3_factors = np.ones((closest_3_factors.shape[0],1))/len(closest_3_factors)
        h_averaged_3 = np.dot(gram_matrix,np.dot(closest_3_pi_matrix,uniform_closest_3_factors))
        averaged_3_est_predictions = h_averaged_3[orphan_indices, :]
        averaged_3_est_predictions = averaged_3_est_predictions[indices, :]
        averaged_3_est_predictions = averaged_3_est_predictions.reshape(1, averaged_3_est_predictions.shape[0])
        averaged_3_rmse = rmse(np.array(y),np.array(averaged_3_est_predictions))
        averaged_3_kt = kendalltau(np.array(y),np.array(averaged_3_est_predictions))[0]
        end = time.time()
        averaged_3_time = end - start

        # -------- Model: Best-Protein -------- 
        start = time.time()
        best_protein = None
        best_protein_rmse = np.inf

        best_kt_protein = None
        best_protein_kt = -1

        for i in range(len(other_proteins)):
            p_other = np.dot(gram_matrix, pi_matrix[:,i])
            p_other = np.squeeze(np.asarray(p_other))
            op_est_predictions = p_other[orphan_indices]
            op_est_predictions = p_other[indices]
            op_est_predictions = op_est_predictions.reshape(1,op_est_predictions.shape[0])
            op_rmse = rmse(np.array(y), np.array(op_est_predictions))
            op_kt = kendalltau(np.array(y), np.array(op_est_predictions))[0]

            if op_rmse < best_protein_rmse:
                best_protein = other_proteins[i]
                best_protein_rmse = op_rmse

            if op_kt > best_protein_kt:
                best_kt_protein = other_proteins[i]
                best_protein_kt = op_kt

        end = time.time()
        best_protein_time = end - start

        # Method for application of the supervised models
        param_grid = {"epsilon": [0.1, 0.01, 0.001], "C": [2 ** i for i in range(-5, 6)]}
        param_tuples = ParameterGrid(param_grid)
        kf = KFold(n_splits=3)

        def get_supervised_rmse_and_kt(validation_indices, test_indices, param_tuples):

            X = protein_XY_dict[orphan]["X"][validation_indices]
            y = protein_XY_dict[orphan]["y"]
            y_val = y[validation_indices]

            protein_gram = gram_matrix[orphan_indices, :]
            protein_gram = protein_gram[:, orphan_indices]
            protein_val_gram = protein_gram[validation_indices, :]
            protein_val_gram = protein_val_gram[:, validation_indices]

            best_rmse = np.inf
            best_parameter = None

            best_kt = -1
            best_parameter_kt = None

            for parameter in param_tuples:
                p_rmse = []
                p_kt = []
                for train_index, test_index in kf.split(X):
                    relevant_gram = protein_val_gram[train_index, :]
                    relevant_gram = relevant_gram[:, train_index]
                    Xids = np.array(orphan_indices)[validation_indices]
                    Xids = Xids[train_index]
                    svs, coefs = svr.svr_train(relevant_gram, Xids, y_val[train_index], parameter["C"], parameter["epsilon"])

                    K_test_svs = gram_matrix[orphan_indices, :]
                    K_test_svs = K_test_svs[validation_indices, :]
                    K_test_svs = K_test_svs[test_index, :]
                    K_test_svs = K_test_svs[:, svs]

                    y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(test_index)))

                    p_rmse.append(rmse(y_pred, y_val[test_index]))
                    p_kt.append(kendalltau(y_pred, y_val[test_index])[0])

                average_rmse = np.average(p_rmse)
                if best_rmse > average_rmse:
                    protein_models[orphan] = (svs, coefs)
                    best_rmse = average_rmse
                    best_parameter = parameter

                average_kt = np.average(p_kt)
                if average_kt > best_kt:
                    best_kt = average_kt
                    best_parameter_kt = parameter

            relevant_gram = protein_gram[validation_indices, :]
            relevant_gram = relevant_gram[:, validation_indices]
            Xids = np.array(orphan_indices)
            Xids = Xids[validation_indices]

            svs, coefs = svr.svr_train(relevant_gram, Xids, y[validation_indices], best_parameter["C"], best_parameter["epsilon"])

            K_test_svs = gram_matrix[orphan_indices, :]
            K_test_svs = K_test_svs[test_indices, :]
            K_test_svs = K_test_svs[:, svs]

            y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(test_indices)))

            super_rmse = rmse(y_pred, y[test_indices])

            svs, coefs = svr.svr_train(relevant_gram, Xids, y[validation_indices], best_parameter_kt["C"], best_parameter_kt["epsilon"])

            K_test_svs = gram_matrix[orphan_indices, :]
            K_test_svs = K_test_svs[test_indices, :]
            K_test_svs = K_test_svs[:, svs]

            y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(test_indices)))

            super_kt = kendalltau(y_pred, y[test_indices])[0]

            return (super_rmse,super_kt)


        # -------- Application of the supervised models --------
        start = time.time() 
        super_05_rmse, super_05_kt = get_supervised_rmse_and_kt(validation_indices_5, test_indices_5,param_tuples)
        end = time.time()
        super_05_time = end - start

        start = time.time() 
        super_10_rmse, super_10_kt = get_supervised_rmse_and_kt(validation_indices_10, test_indices_10,param_tuples)
        end = time.time()
        super_10_time = end - start

        start = time.time() 
        super_30_rmse, super_30_kt = get_supervised_rmse_and_kt(validation_indices_30, test_indices_30,param_tuples)
        end = time.time()
        super_30_time = end - start

        start = time.time() 
        super_50_rmse, super_50_kt = get_supervised_rmse_and_kt(validation_indices_50, test_indices_50,param_tuples)
        end = time.time()
        super_50_time = end - start

        start = time.time() 
        super_80_rmse, super_80_kt = get_supervised_rmse_and_kt(validation_indices_80, test_indices_80,param_tuples)
        end = time.time()
        super_80_time = end - start

        # -------- Model: Target-Ligand Kernel -------- 
        '''
        start = time.time() 
        # --- Building K_X for TLK
        #print("Calculating TLK")
        non_orphan_indices = np.array([order_per_protein[target_protein] for target_protein in other_proteins]).flatten()
        orphan_indices = orphan_indices
        K_X = gram_matrix
        #K_X = K_X[:,non_orphan_indices]

        K_T = None
        for t1 in proteins:
            K_t1 = None
            for t2 in proteins:
                if K_t1 is None:
                    # minLines equals the amount of ligands per protein
                    K_t1 = protein_sims[t1][t2] * np.ones((minLines,minLines))
                else:
                    K_t1 = np.hstack([K_t1,protein_sims[t1][t2] * np.ones((minLines,minLines))])
            if K_T is None:
                K_T = K_t1
            else:
                K_T = np.vstack([K_T,K_t1])

        K_TL = np.multiply(K_X,K_T)

        y_s = np.array([protein_XY_dict[target]["y"] for target in proteins]).flatten()
        y_train = y_s[non_orphan_indices]

        # --- Train SVR with K_TL
        best_rmse = np.inf
        best_parameter = None

        best_kt = -1
        best_parameter_kt = None
        print("Determining best parameters for TLK")
        for parameter in param_tuples:
            p_rmse = []
            p_kt = []
            for train_index, test_index in kf.split(non_orphan_indices):
                relevant_gram = K_TL[non_orphan_indices,:]
                relevant_gram = relevant_gram[:,non_orphan_indices]
                relevant_gram = relevant_gram[train_index,:]
                relevant_gram = relevant_gram[:,train_index]
                Xids = non_orphan_indices
                Xids = Xids[train_index]
                svs, coefs = svr.svr_train(relevant_gram, Xids , y_train[train_index], parameter["C"], parameter["epsilon"])

                K_test_svs = K_TL[non_orphan_indices, :]
                K_test_svs = K_test_svs[test_index,:]
                K_test_svs= K_test_svs[:,svs]

                y_pred = np.array(svr.predict(K_test_svs,np.array(coefs).T)).reshape((len(test_index)))

                p_rmse.append(rmse(y_pred, y_train[test_index]))
                p_kt.append(kendalltau(y_pred,y_train[test_index])[0])

            average_rmse = np.average(p_rmse)
            if best_rmse > average_rmse:
                #print("Current best RMSE for TLK: {}".format(average_rmse))
                #protein_models[protein] = (svs,coefs)
                best_rmse = average_rmse
                best_parameter = parameter

            average_kt = np.average(p_kt)
            if average_kt > best_kt:
                #print("Current best KT for TLK: {}".format(average_kt))
                #protein_models[protein] = (svs,coefs)
                best_kt = average_kt
                best_parameter_kt = parameter

        relevant_gram = K_TL[non_orphan_indices, :]
        relevant_gram = relevant_gram[:, non_orphan_indices]
        Xids = np.array(non_orphan_indices)

        svs, coefs = svr.svr_train(relevant_gram, Xids, y_train, best_parameter["C"], best_parameter["epsilon"])

        K_test_svs = K_TL[orphan_indices, :]
        K_test_svs = K_test_svs[indices, :]
        K_test_svs = K_test_svs[:, svs]

        y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(indices)))
        y_orphan_test = y_s[orphan_indices]
        y_orphan_test = y_orphan_test[indices]
        TLK_rmse = rmse(y_pred, y_orphan_test)

        svs, coefs = svr.svr_train(relevant_gram, Xids, y_train, best_parameter_kt["C"], best_parameter_kt["epsilon"])

        K_test_svs = K_TL[orphan_indices, :]
        K_test_svs = K_test_svs[indices, :]
        K_test_svs = K_test_svs[:, svs]

        y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(indices)))
        y_orphan_test = y_s[orphan_indices]
        y_orphan_test = y_orphan_test[indices]
        TLK_kt = kendalltau(y_pred, y_orphan_test)[0]

        end = time.time()
        TLK_time = end - start


        # -------- Model: Target-Ligand Kernel - Closest 3 -------- 

        start = time.time()
        # --- Building K_X for TLK - Closest 3
        #print("Calculating TLK for Closest 3 only")
        non_orphan_proteins = closest_3_proteins
        non_orphan_indices = np.array([order_per_protein[target_protein] for target_protein in non_orphan_proteins]).flatten()
        K_X = gram_matrix
        # K_X = K_X[:,non_orphan_indices]

        K_T = None
        for t1 in proteins:
            K_t1 = None
            for t2 in proteins:
                if K_t1 is None:
                    # minLines equals the amount of ligands per protein
                    K_t1 = protein_sims[t1][t2] * np.ones((minLines, minLines))
                else:
                    K_t1 = np.hstack([K_t1, protein_sims[t1][t2] * np.ones((minLines, minLines))])
            if K_T is None:
                K_T = K_t1
            else:
                K_T = np.vstack([K_T, K_t1])

        K_TL = np.multiply(K_X, K_T)

        y_s = np.array([protein_XY_dict[target]["y"] for target in proteins]).flatten()
        y_train = y_s[non_orphan_indices]

        # --- Train SVR with K_TL
        best_rmse = np.inf
        best_parameter = None

        best_kt = -1
        best_parameter_kt = None
        #print("Determining best parameters for TLK")
        for parameter in param_tuples:
            p_rmse = []
            p_kt = []
            for train_index, test_index in kf.split(non_orphan_indices):
                relevant_gram = K_TL[non_orphan_indices, :]
                relevant_gram = relevant_gram[:, non_orphan_indices]
                relevant_gram = relevant_gram[train_index, :]
                relevant_gram = relevant_gram[:, train_index]
                Xids = non_orphan_indices
                Xids = Xids[train_index]
                svs, coefs = svr.svr_train(relevant_gram, Xids, y_train[train_index], parameter["C"], parameter["epsilon"])

                K_test_svs = K_TL[non_orphan_indices, :]
                K_test_svs = K_test_svs[test_index, :]
                K_test_svs = K_test_svs[:, svs]

                y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(test_index)))

                p_rmse.append(rmse(y_pred, y_train[test_index]))
                p_kt.append(kendalltau(y_pred, y_train[test_index])[0])

            average_rmse = np.average(p_rmse)
            if best_rmse > average_rmse:
                #print("Current best RMSE for TLK: {}".format(average_rmse))
                #protein_models[protein] = (svs,coefs)
                best_rmse = average_rmse
                best_parameter = parameter

            average_kt = np.average(p_kt)
            if average_kt > best_kt:
                #print("Current best KT for TLK: {}".format(average_kt))
                #protein_models[protein] = (svs,coefs)
                best_kt = average_kt
                best_parameter_kt = parameter

        relevant_gram = K_TL[non_orphan_indices, :]
        relevant_gram = relevant_gram[:, non_orphan_indices]
        Xids = np.array(non_orphan_indices)

        svs, coefs = svr.svr_train(relevant_gram, Xids, y_train, best_parameter["C"], best_parameter["epsilon"])

        K_test_svs = K_TL[orphan_indices, :]
        K_test_svs = K_test_svs[indices, :]
        K_test_svs = K_test_svs[:, svs]

        y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(indices)))
        y_orphan_test = y_s[orphan_indices]
        y_orphan_test = y_orphan_test[indices]
        TLK_clo3_rmse = rmse(y_pred, y_orphan_test)

        svs, coefs = svr.svr_train(relevant_gram, Xids, y_train, best_parameter_kt["C"], best_parameter_kt["epsilon"])

        K_test_svs = K_TL[orphan_indices, :]
        K_test_svs = K_test_svs[indices, :]
        K_test_svs = K_test_svs[:, svs]

        y_pred = np.array(svr.predict(K_test_svs, np.array(coefs).T)).reshape((len(indices)))
        y_orphan_test = y_s[orphan_indices]
        y_orphan_test = y_orphan_test[indices]
        TLK_clo3_kt = kendalltau(y_pred, y_orphan_test)[0]

        end = time.time()
        TLK_clo3_time = end - start
        '''
        
        # ------------------------- Results -------------------------

        print("-------------- Test Set Performance RMSE: --------------")
        print("NLCP: {}".format(nlcp_rmse))
        print("OPCA: {}".format(opca_rmse))
        print("Simplified: {}".format(simplified_rmse))
        #print("TLK: {}".format(TLK_rmse))
        #print("TLK-Clo-3: {}".format(TLK_clo3_rmse))
        print("Averaged: {}".format(averaged_rmse))
        print("Averaged-CLo-3: {}".format(averaged_3_rmse))
        print("Closest Protein {}: {}".format(closest_protein,closest_rmse))
        print("Farthest Protein {}: {}".format(farthest_protein,farthest_rmse))
        print("Best Protein {}: {}".format(best_protein,best_protein_rmse))
        print("Super-5: {}".format(super_05_rmse))
        print("Super-10: {}".format(super_10_rmse))
        print("Super-30: {}".format(super_30_rmse))
        print("Super-50: {}".format(super_50_rmse))
        print("Super-80: {}".format(super_80_rmse))

        print("-------------- Test Set Performance KT: --------------")
        print("NLCP: {}".format(nlcp_kt))
        print("OPCA: {}".format(opca_kt))
        print("Simplified: {}".format(simplified_kt))
        #print("TLK: {}".format(TLK_kt))
        #print("TLK-Clo-3: {}".format(TLK_clo3_kt))
        print("Averaged: {}".format(averaged_kt))
        print("Averaged-CLo-3: {}".format(averaged_3_kt))
        print("Closest Protein {}: {}".format(closest_protein,closest_kt))
        print("Farthest Protein {}: {}".format(farthest_protein,farthest_kt))
        print("Best Protein {}: {}".format(best_protein,best_protein_kt))
        print("Super-5: {}".format(super_05_kt))
        print("Super-10: {}".format(super_10_kt))
        print("Super-30: {}".format(super_30_kt))
        print("Super-50: {}".format(super_50_kt))
        print("Super-80: {}".format(super_80_kt))

        print("-------------- Test Set Performance - Execution Time (seconds): --------------")
        print("NLCP: {}".format(nlcp_time))
        print("OPCA: {}".format(opca_time))
        print("Simplified: {}".format(simplified_time))
        #print("TLK: {}".format(TLK_time))
        #print("TLK-Clo-3: {}".format(TLK_clo3_time))
        print("Averaged: {}".format(averaged_time))
        print("Averaged-CLo-3: {}".format(averaged_3_time))
        print("Closest Protein {}: {}".format(closest_protein,closest_time))
        print("Farthest Protein {}: {}".format(farthest_protein,farthest_time))
        print("Best Protein {}: {}".format(best_protein,best_protein_time))
        print("Super-5: {}".format(super_05_time))
        print("Super-10: {}".format(super_10_time))
        print("Super-30: {}".format(super_30_time))
        print("Super-50: {}".format(super_50_time))
        print("Super-80: {}".format(super_80_time))

        rmse_file.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(orphan,nlcp_rmse,opca_rmse,simplified_rmse,averaged_rmse,averaged_3_rmse,closest_rmse,farthest_rmse,best_protein_rmse,super_05_rmse,super_10_rmse,super_30_rmse,super_50_rmse,super_80_rmse))
        kt_file.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(orphan,nlcp_kt,opca_kt,simplified_kt,averaged_kt,averaged_3_kt,closest_kt,farthest_kt,best_protein_kt,super_05_kt, super_10_kt, super_30_kt, super_50_kt, super_80_kt))
        time_file.write("{};{};{};{};{};{};{};{};{};{};{};{};{};{}\n".format(orphan,nlcp_time,opca_time,simplified_time,averaged_time,averaged_3_time,closest_time,farthest_time,best_protein_time,super_05_time, super_10_time, super_30_time, super_50_time, super_80_time))


    rmse_file.close()
    kt_file.close()
    time_file.close()

# To check for errors in the matrix created in the optimization
print("NaN values:", nan_values)


import pandas as pd
import os
import numpy as np
from scipy import sparse
from numpy.random import choice, seed
import collections
import pickle

### DATA PREPARATION AND SETUP ###

max_entry = 0 # Variable used to determine the maximum amount of features (summing all the compounds of the protein)
minLines = 240 # How many lines/compounds should at least be present per protein to consider it?
maxProteins = 30
undersampling = True # Should we undersample proteins with more than 90 lines?

number_draws = 10
randomseed = 0

double_check = True


index_column_name = "UniprotId"
problematic_compounds = set(['261646', '337945', '65096', '117670', '52940', '71188', '69818', '336015', '309971', '253037', '180446', '597273', '456583', '175291', '108057', '1068100', '76692', '241697', '58705', '294115', '50801', '12067', '29750', '30512', '601862', '12667', '249104', '1079958', '279506', '348270', '107819', '435901', '242932', '283249', '592217', '161859', '437594', '31275', '148300', '566442', '595285', '276843', '286544', '130084', '118340', '1107650', '209816', '305939', '48576', '183381', '82530', '108873', '287060', '226613', '591435', '222984', '13867', '178739', '75207', '328648', '67816', '117322', '154296', '286437', '1086651', '108277', '13334', '3093', '1109953', '154278', '940252', '319217', '58673', '259468', '580383', '7342', '47360', '256557', '446758', '223446', '1078359', '211656', '230606', '1104016', '1104017', '1103048', '19054', '334544', '80574', '105541', '224111', '134462', '107828', '398870', '516124', '107822', '617540', '1104453', '45387', '1109176', '123568', '49413', '1104455', '154588', '1104458', '348194', '1087111', '69518', '68893', '63200', '118604', '106620', '92761', '41420', '41803', '228796', '69450', '70481', '178928', '136916', '58594', '608212', '45940', '197670', '360018', '504113', '415008', '264852', '31286', '293483', '154320', '31694', '633315', '1064311', '75087', '592387', '583812', '13372', '178853', '439693', '298729', '336335', '24309', '1085841', '1101432', '332303', '403088', '81653', '394633', '581045', '595088', '579438', '70590', '299270', '947202', '105570', '293156', '152193', '10699', '153167', '75151', '87573', '27681', '63799', '1086608', '186437', '76270', '129526', '398027', '321380', '584664', '319069', '286934', '1108506', '252801', '58145', '10652', '1078985', '1103660', '389172', '613472', '335122', '127562', '206805', '48197', '249947', '340556', '574653', '593456', '272713', '257995', '217612', '10238', '151803', '163861', '602613', '258331', '1104454', '278719', '26697', '178815', '67244', '257688', '263021', '326235', '256670', '611234', '346913', '1107640', '215822', '205388', '62116', '81347', '586066', '135295', '99608', '347198', '227415', '72000', '277704', '258050', '144467', '134470', '63194', '47774', '692138', '22513', '64185', '4341', '116759', '46481', '1107639', '225348', '182973', '263410', '356167', '134372', '456734', '617853', '154280', '504152', '946650', '245600', '79409', '47376', '519309', '273663', '48380', '352920', '251389', '155105', '39985', '598226', '165150', '639612', '587945', '1086636', '1086634', '1086633', '601515', '1085842', '286960', '264668', '71296', '275106', '84564', '588907', '11525', '287041', '287165', '148178', '132263', '1063456', '309954', '333695', '599158', '1088013', '1088012', '1104634', '615885', '244822', '161697', '25008', '255432', '77664', '959799', '959798', '612920', '121604', '647583', '185752', '333017', '65951', '347707', '118535', '79634', '36267', '283854', '341605', '345202', '261089', '616272', '332568', '602941', '391459', '565834', '388661', '107453', '70563', '1086383', '218222', '1086387', '1086386', '1086384', '303059', '1086389', '1086388', '84145', '10070', '390652', '438311', '1086629', '211372', '313105', '180687', '135335', '1078222', '49997', '254618', '304846', '519355', '63509', '251382', '609421', '10437', '106118', '241931', '600708', '107945', '1078375', '106117', '131866', '280218', '251472', '321049', '43856', '612962', '334421', '1104456', '259742', '156390', '334745', '264449', '201146', '598539', '597404', '341232', '215923', '442110', '81699', '295515', '293738', '335256', '183803', '183806', '263711', '512524', '604053', '10915', '121943', '250172', '347003', '46361', '263478', '337543', '598981', '584348', '107029', '143980', '1107624', '24504', '598623', '132297', '72928', '154426', '1080562', '63374', '229679', '507040', '229401', '254491', '245444', '255991', '231262', '423013', '303759', '256573', '80399', '62962', '70525', '606322', '1086625', '601054', '1103661', '254014', '125128', '245232', '171274', '357828', '246617', '156331', '32686', '69828', '583120', '96882', '563407', '1086267', '576563', '65039', '176346', '133554', '123682', '224893', '591593', '145120', '265876', '572585', '25374', '106707', '262690', '224564', '323985', '323984', '230430', '230701', '10958', '1079942', '213268', '266389', '598112', '347049', '229620', '339802', '244151', '25512', '593336', '439114', '338731', '311844', '107901', '58542', '1078115', '283550', '290268', '245129', '180780', '6453', '336234', '63035', '211243', '103770', '19300', '163027', '27662', '180779', '444360', '6067', '250985', '574839', '117642', '6436', '276422', '1078203', '943668', '69865', '260107', '1062469', '589469', '8959', '245857', '340525', '249960', '335140', '70776', '221402', '278078', '106704', '2922', '1104449', '72730', '69269', '77433', '49879', '226821', '463261', '604203', '1087828', '596123', '220666', '275356', '150031', '226796', '150032', '238145', '333018', '251815', '301028', '247651', '54369', '335486', '560889', '270636', '108792', '79079', '250498', '69609', '567427', '183351', '584850', '161933', '107377', '257842', '6492', '154234', '71946', '144798', '40674', '331177', '1088391', '281662', '130049', '591270', '1086505', '1086504', '594876', '1065254', '1086503', '1086502', '64520', '1064379', '588662', '47725', '1065028', '135402', '270243', '217190', '513209', '692477', '337542', '154806', '23625', '328290', '1108492', '1106674', '1104450', '183303', '329589', '245059', '105591', '80015', '1104457'])

def readSimilarityFile(filename, index_column_name):
    '''
    Read in Similarity files to obtain similarities among proteins.
    '''
    return pd.read_csv(filename, sep=",", index_col=index_column_name)

def readBindingsToFeatureDict(folder, max_entry, minLines, maxProteins, undersampling):
    '''
    Reads in the samples per protein, takes the maximum feature number as amount of features.
    Creates a sparse matrix representation and stores the features X as well as the label
    y in a dictionary.

    :param folder:
    :return: Dictionary of features X and label y
    '''

    global randomseed
    temp_dict = {} # Store feature vectors and labels
    compound_to_index = {}  # Assigns global index to each compound name
    max_compound = 0 # Global index assigned to a particular compound
    protein_index_to_compound = {}  # Assigns compound name to each line_num of a particular protein (compound present in the X position of a combination)
    compound_feature = {}  # Assigns feature vector to the global index of a particular compound
    order_per_protein = {}  # Assigns global compound index to each line of a particular protein (order of the compounds in a combination)

    used_compounds_set = set([]) # Used to check whether a particular component has already been used in the combination

    target_compound_sets = {} # List of compounds used in a particular protein

    # Take all CSV file from folder
    files = [file for file in os.listdir(folder) if ".csv" in file]

    # ------------------------- Iterate over CSV files -------------------------

    file_length_dict_tuples = []

    # Sort the files according to length for the subsampling
    for file in files:
        with open(folder + "/" + file, "r") as f:
            file_length_dict_tuples.append((file,len(f.readlines())))

    sorted_files = sorted(file_length_dict_tuples, key=lambda x: x[1])
    sorted_files = [file[0] for file in sorted_files]
    print("Sorted files:", sorted_files)

    # ------------ Combination process ------------

    print("------------------------ Combination process ------------------------")

    protein_count = 0 # To check the maximum of proteins considered

    for file in sorted_files:
        print("---------- Protein count:", protein_count, "----------")
        with open(folder + "/" + file, "r") as f:
            lines = f.readlines()
            lines = [line for line in lines if line.split(",")[0].strip() not in problematic_compounds]

            # What is the maximum feature number, as seen in the header?
            max_in_file = int(str(lines[0]).split(",")[-2].strip().split("=")[1])

            # Is the maximum feature number higher than current max?
            if max_in_file > max_entry:
                max_entry = max_in_file

            # Check whether more than minLines lines/compounds are available (MORE THAN because 1st line is header)
            # Already without the problematic_compounds
            if len(lines) > minLines:
                unused_compound_lines = [] # List of compounds extracted from the particular file

                # Check which compounds have already been used to keep the set distinct
                for line in lines[1:]:
                    line = str(line)
                    l_split = line.split(",")
                    compound_name = l_split[0].strip()
                    if(compound_name not in used_compounds_set) or (double_check is False):
                        unused_compound_lines.append(line)

                # If the number of components is less than the minLines (we want to have enough variability)
                if len(unused_compound_lines) >= minLines:
                    if protein_count >= maxProteins:
                        continue # continue statement returns the control to the beginning of the for/while loop

                    protein_count += 1

                    # Get the protein name from the file name
                    protein_name = file.split("_")[0]
                    order_per_protein[protein_name] = []

                    print("---------- Protein name:", protein_name, "----------")

                    target_compound_sets[protein_name] = set([])

                    # Used to store feature vector and label
                    temp_dict[protein_name] = []

                    # If undersampling is on, randomly draw lines from a file
                    #@TODO: Should the seed be set in the beginning? As of now we always take the same lines for each file
                    print("Enough lines: {}".format(len(lines)))
                    if undersampling:
                        seed(randomseed)
                        randomseed += 1
                        # Compound combination process for a particular protein
                        selectedLines = choice(unused_compound_lines, minLines, replace=False)
                    else:
                        selectedLines = unused_compound_lines

                    print("Selected lines size: {}".format(len(selectedLines)))

                    protein_index_to_compound[protein_name] = {}

                    line_num = 0 # Order of the compound in the combination
                    
                    # Obtain features and label from each line
                    for line in selectedLines:
                        line = str(line)
                        l_split = line.split(",")
                        compound_name = l_split[0].strip()
                        if compound_name not in compound_to_index:
                            compound_to_index[compound_name] = max_compound
                            max_compound += 1

                        # Establishment of the compound combination (without the feature vectors)
                        protein_index_to_compound[protein_name][line_num] = compound_name
                        line_num += 1 
                        order_per_protein[protein_name] += [compound_to_index[compound_name]]
                        x = list(map(int, l_split[1:-1])) # feature vector
                        y = float(l_split[-1]) # label (affinity)
                        temp_dict[protein_name].append((x, y))
                        used_compounds_set.add(compound_name)
                        target_compound_sets[protein_name].add(compound_name)

                else:
                    print("Not enough lines after duplicate compound removal: {}".format(len(unused_compound_lines)))
            else:
                print("Not enough lines: {}".format(len(lines)))

    print("------------------------ Data structures ------------------------")

    feature_dict = {}

    # Build data structures using a sparse matrix
    for protein in temp_dict:
        print("--------------- Protein:", protein, "---------------")
        feature_dict[protein] = {}
        row = []
        column = []
        data = []
        y = []
        row_ind = 0
        # Next process (for): Protein created from the new combination of compounds
        for tuple in temp_dict[protein]:
            # tuple[0] is the feature vector in a certain row
            # For each row there are len(tuple[0]) positive entries
            row += [row_ind] * len(tuple[0])

            # Each entry from tuple[0] is 1
            data += [1] * len(tuple[0])

            # Columns correspond to features
            column += [entry - 1 for entry in tuple[0]]

            # tuple[1] is the label
            y += [tuple[1]]

            row_ind += 1

        print("Data: {}, Rows: {}, Columns: {}".format(len(data), len(row), len(column)))
        print("Different valued rows (maximum of compounds used):", row_ind)
        print("Number of columns (maximum of features used):", max_entry)
        print("Highest positive value (from feature values):", max(column))

        # Create a sparse matrix with the data (row, column)
        # We lost some features at the end of the columns, because sometimes max_entry < len(column)
        X = sparse.csr_matrix((data, (row, column)), shape=(row_ind, max_entry))
        print("Sparse matrix shape (number of compounds, number of features):", X.toarray().shape)

        # Establishment of the compound combination (now with the feature vectors, and the affinity after)
        # Assign each feature vector to the respective compound global index, for all compounds in the created protein (however, this compound_feature dict is not associated with any protein)
        for i in range(X.shape[0]):
            compound_feature[compound_to_index[protein_index_to_compound[protein][i]]] = X.getrow(i)
        y = np.array(y)

        # Store feature vector and label (affinity) per protein
        feature_dict[protein]["X"] = X
        feature_dict[protein]["y"] = y

    return feature_dict, compound_to_index, protein_index_to_compound, compound_feature, target_compound_sets, order_per_protein

# --------------------------------------- Main ---------------------------------------

folder = "../data"

# Read protein similarities
protein_sims = readSimilarityFile(folder+"/similarityMatrices/refinedTargetsPrctIdent.csv", index_column_name, )

# Create experiments folder
experiments_folder = folder+"/fixed_datasets/"+"minLines_"+str(minLines)+"_undersampling_"+str(undersampling)+"_numberDraws_"+str(number_draws)

if not os.path.isdir(experiments_folder):
    os.mkdir(experiments_folder)

# Read compound bindings (number_draws times)
for i in range(10):

    print("------------------------ Draw", i, "------------------------")

    draw_folder = experiments_folder+"/"+str(i)

    if not os.path.isdir(draw_folder):
        os.mkdir(draw_folder)

    # Build the feature and label dict for all proteins
    protein_XY_dict, compound_to_index, protein_index_to_compound, compound_feature, target_compound_sets, order_per_protein = readBindingsToFeatureDict(folder+"/zipped_uniform_ecfp4/ecfp4",  max_entry, minLines, maxProteins, undersampling)

    with open(draw_folder+"/protein_XY_dict.p","wb") as f:
        pickle.dump(protein_XY_dict,f)

    with open(draw_folder+"/compound_to_index.p","wb") as f:
        pickle.dump(compound_to_index, f)

    with open(draw_folder+"/protein_index_to_compound.p","wb") as f:
        pickle.dump(protein_index_to_compound,f)

    with open(draw_folder+"/compound_feature.p","wb") as f:
        pickle.dump(compound_feature,f)

    with open(draw_folder+"/target_compound_sets.p","wb") as f:
        pickle.dump(target_compound_sets,f)

    with open(draw_folder+"/order_per_protein.p","wb") as f:
        pickle.dump(order_per_protein,f)

    with open(draw_folder+"/max_entry.p","wb") as f:
        pickle.dump(max_entry,f)

    with open(draw_folder+"/protein_sims.p","wb") as f:
        pickle.dump(protein_sims,f)

    with open(draw_folder+"/minLines.p","wb") as f:
        pickle.dump(minLines,f)

    with open(draw_folder+"/double_check.p","wb") as f:
        pickle.dump(double_check,f)

    with open(draw_folder + "/maxProteins.p", "wb") as f:
        pickle.dump(maxProteins, f)

    # Sanity check whether list of compounds used per target is disjunct
    target_compound_intersect = np.array([list(compounds) for compounds in [target_compound_sets[target] for target in target_compound_sets]]).flatten()
    target_compound_intersect = [item for item, count in collections.Counter(target_compound_intersect).items() if count > 1]

    print("For draw {} overlapping compounds over all proteins are: {}".format(number_draws,len(target_compound_intersect)))

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun


__author__ = "Łukasz Struski"


# ____________________MAIN FUNCTION_____________________#


def main():
    # if len(sys.argv) > 2:
    #     path_dir_data = sys.argv[1]
    #     type = sys.argv[2]+"_"
    #     dataname = path_dir_data[2:-1]
    # else:
    #     path_dir_data = sys.argv[1]
    #     type = ""
    path_dir_data = sys.argv[1]
    dataname = path_dir_data[2:-1]
    

    # parameters for SVM
    C = 1
    gamma = 1.e-3

    precomputed_svm = SVC(C=C, kernel='precomputed')


    X_train = np.genfromtxt(os.path.join(path_dir_data, 'train_data.txt'), dtype=float, delimiter=',')
    y_train = np.genfromtxt(os.path.join(path_dir_data, 'train_labels.txt'), dtype=float, delimiter=',')
    X_test = np.genfromtxt(os.path.join(path_dir_data, 'test_data.txt'), dtype=float, delimiter=',')
    y_test = np.genfromtxt(os.path.join(path_dir_data, 'test_labels.txt'), dtype=float, delimiter=',')

    
    index_train = np.arange(X_train.shape[0])
    index_test = np.arange(X_test.shape[0]) + X_train.shape[0]
    X = np.concatenate((X_train, X_test), axis=0)
    #del X_train, X_test

    index_train = index_train.astype(np.intc)
    index_test = index_test.astype(np.intc)
    acc_genRBF = []
    f1_genRBF = []
    for type in ["zero","mice","mean"]:
        type = type+"_"
        # read data
        m = np.genfromtxt(os.path.join(path_dir_data, type + 'mu.txt'), dtype=float, delimiter=',')
        cov = np.genfromtxt(os.path.join(path_dir_data, type + 'cov.txt'), dtype=float, delimiter=',')

        # train
        rbf_ker = rbf.RBFkernel(m, cov, X)
        S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                        rbf_ker.S)
        
        # S_train, S_test, completeDataId_train, completeDataId_test = rbf.trainTestID_1(index_test, index_train,
        #                                                                             rbf_ker.S)


        S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)


        train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)
        precomputed_svm.fit(train, y_train)

        # test
        S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
        test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                                completeDataId_test_new, completeDataId_train_new)

        y_pred = precomputed_svm.predict(test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print("genRBF Accuracy classification score: {:.2f}".format(acc))
        acc_genRBF.append(acc)
        f1_genRBF.append(f1)



    mean_imputer = SimpleImputer(strategy='mean')
    X_train_mean_imputed = mean_imputer.fit_transform(X_train)
    X_test_mean_imputed = mean_imputer.transform(X_test)

    svm = SVC(C=C,kernel='rbf')
    svm.fit(X_train_mean_imputed, y_train)
    y_pred = svm.predict(X_test_mean_imputed)
    acc_mean_imputer = accuracy_score(y_test, y_pred)
    f1_mean_imputer = f1_score(y_test, y_pred, average='macro')
    print("Mean Imputer Accuracy classification score: {:.2f}".format(acc_mean_imputer))


    # Impute missing values using MICE
    mice_imputer = IterativeImputer()
    X_train_mice_imputed = mice_imputer.fit_transform(X_train)
    X_test_mice_imputed = mice_imputer.transform(X_test)

    svm = SVC(C=C,kernel='rbf')
    svm.fit(X_train_mice_imputed, y_train)
    y_pred = svm.predict(X_test_mice_imputed)
    acc_mice_imputer = accuracy_score(y_test, y_pred)
    f1_mice_imputer = f1_score(y_test, y_pred, average='macro')
    print("MICE Imputer Accuracy classification score: {:.2f}".format(acc_mice_imputer))

    score_dict = {
        "genRBF zero": [acc_genRBF[0], f1_genRBF[0]],
        "genRBF mice": [acc_genRBF[1], f1_genRBF[1]],
        "genRBF mean": [acc_genRBF[2], f1_genRBF[2]],
        "Mean Imputer": [acc_mean_imputer, f1_mean_imputer],
        "MICE Imputer": [acc_mice_imputer, f1_mice_imputer]
    }

    # Create a DataFrame from the score dictionary
    df = pd.DataFrame.from_dict(score_dict, orient='index', columns=['Accuracy Score', 'F1 Score'])

    # Round F1 score to 5 decimal places
    df['F1 Score'] = df['F1 Score'].round(5)
    df['Accuracy Score'] = df['Accuracy Score'].round(5)

    # Define the CSV file path
    csv_file = "results/{}.csv".format(dataname)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file, index_label='Imputer')

if __name__ == "__main__":
    main()

    pass

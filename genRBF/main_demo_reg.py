import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from genRBF_source import RBFkernel as rbf
from genRBF_source import cRBFkernel as fun
from load_txt import load_data




def main():
    dataname, mech, para = sys.argv[1], sys.argv[2], sys.argv[3]

    # parameters for SVM
    C = 1
    gamma = 1.e-3

    precomputed_svm = SVR(C=C, kernel='precomputed')

    #X, y, index, prefilled_path = load_txt(dataname, types, para, full_norm)
    X,y, index = load_data(dataname,mech,para)
    acc_genRBF = []
    f1_genRBF = []

    for fill_type in ["zero","mean","mice"]:

        acc_cv = []
        f1_cv = []

        for fold_n in index.keys():
            fold = index[fold_n]

            train_index = fold['train_index']
            test_index = fold['test_index']

            X_train = X[train_index].astype(np.float64)
            X_test = X[test_index].astype(np.float64)

            y_train = y[train_index]
            y_test = y[test_index]
        
            index_train = np.arange(X_train.shape[0])
            index_test = np.arange(X_test.shape[0])

            #X = np.concatenate((X_train, X_test), axis=0)
            #del X_train, X_test

            index_train = index_train.astype(np.intc)
            index_test = index_test.astype(np.intc)



            # read data
            prefilled_path = f'prefilled_data/{fill_type}/{dataname}/{mech}/'
            m = np.genfromtxt(os.path.join(f'{prefilled_path}/{para}_mu.txt'), dtype=float, delimiter=',')
            cov = np.genfromtxt(os.path.join(f'{prefilled_path}/{para}_cov.txt'), dtype=float, delimiter=',')


            #print(m.dtype,cov.dtype,X_train.dtype)
            # train
            rbf_ker = rbf.RBFkernel(m, cov, X_train)
            S_train, S_test, completeDataId_train, completeDataId_test = fun.trainTestID_1(index_test, index_train,
                                                                                            rbf_ker.S)
            
            S_train_new, completeDataId_train_new = fun.updateSamples(index_train, S_train, completeDataId_train)


            train = rbf_ker.kernelTrain(gamma, index_train, S_train_new, completeDataId_train_new)
            precomputed_svm.fit(train, y_train)

            # test
            S_test_new, completeDataId_test_new = fun.updateSamples(index_test, S_test, completeDataId_test)
            test = rbf_ker.kernelTest(gamma, index_test, index_train, S_test_new, S_train_new,
                                    completeDataId_test_new, completeDataId_train_new)


            y_pred = precomputed_svm.predict(test)
            # print(y_test)
            print(y_pred)
            acc = mean_squared_error(y_test, y_pred)
            f1 = r2_score(y_test, y_pred)

            acc_cv.append(acc)
            f1_cv.append(f1)

            
        
        acc_genRBF.append(np.mean(acc_cv))
        f1_genRBF.append(np.mean(f1_cv))

    score_dict = {
        "zero": [acc_genRBF[0], f1_genRBF[0]],
        "mean": [acc_genRBF[1], f1_genRBF[1]],
        "mice": [acc_genRBF[2], f1_genRBF[2]],

    }

    # Create a DataFrame from the score dictionary
    df = pd.DataFrame.from_dict(score_dict, orient='index',columns=['RMSE Score', 'R2 Score'])

    # Round R2 RMSE score to 5 decimal places
    df['R2 Score'] = df['R2 Score'].round(5)
    df['RMSE Score'] = df['RMSE Score'].round(5)

    # Define the CSV file path
    csv_file_name = "results/{}/{}/{}/{}.csv".format("genRBF", dataname, mech, para)

    # Check if the directory exists, and if not, create it
    directory = os.path.dirname(csv_file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_name, index_label='Imputer')
    print(df)

if __name__ == "__main__":
    main()

    pass

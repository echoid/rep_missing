import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.svm import SVR
from load_txt import load_data
from sklearn.decomposition import PCA, KernelPCA
from ik import IKSimilarity,IKFeature,Isolation_Kernal
from sklearn.neighbors import KDTree

# ____________________MAIN FUNCTION_____________________#

## Only suitable for small amount of missing

def main():
    dataname, mech, para = sys.argv[1], sys.argv[2], sys.argv[3]

    # parameters for SVM
    C = 1
    gamma = 1.e-3

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

            if fill_type == "mean":

                mean_imputer = SimpleImputer(strategy='mean')
                X_train_imputed = mean_imputer.fit_transform(X_train)
                X_test_imputed = mean_imputer.transform(X_test)

            elif fill_type == "mice":
                # Impute missing values using MICE
                mice_imputer = IterativeImputer()
                X_train_imputed = mice_imputer.fit_transform(X_train)
                X_test_imputed = mice_imputer.transform(X_test)



            elif fill_type == "zero":
                zero_imputer = SimpleImputer(strategy='constant', fill_value=0)
                X_train_imputed = zero_imputer.fit_transform(X_train)
                X_test_imputed = zero_imputer.transform(X_test)


            IK = Isolation_Kernal(psi=128, t=200,KD_tree=False)


            IK.build(X_train_imputed)
            train_feature = IK.generate_feature(X_train_imputed)
            test_feature = IK.generate_feature(X_test_imputed)
            test_sim = IK.similarity(test_feature,train_feature)
            
            train_sim = IK.similarity(train_feature,train_feature)
            

            svm = SVR(C=C, kernel='precomputed')
            svm.fit(train_sim, y_train)

            y_pred = svm.predict(test_sim)
            acc = mean_squared_error(y_test, y_pred)
            f1 = r2_score(y_test, y_pred)



            acc_cv.append(acc)
            f1_cv.append(f1)



        acc_genRBF.append(np.mean(acc_cv))
        f1_genRBF.append(np.mean(f1_cv))

    score_dict = {
        "zero": [acc_genRBF[0], f1_genRBF[0]],
        "mean": [acc_genRBF[0], f1_genRBF[0]],
        "mice": [acc_genRBF[1], f1_genRBF[1]],

    }

    # Create a DataFrame from the score dictionary
    df = pd.DataFrame.from_dict(score_dict, orient='index',columns=['RMSE Score', 'R2 Score'])

    # Round R2 RMSE score to 5 decimal places
    df['R2 Score'] = df['R2 Score'].round(5)
    df['RMSE Score'] = df['RMSE Score'].round(5)




    # Define the CSV file path
    csv_file_name = "results/{}/{}/{}/{}.csv".format("IK", dataname, mech, para)

    # Check if the directory exists, and if not, create it
    directory = os.path.dirname(csv_file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_name, index_label='Imputer')
if __name__ == "__main__":
    main()

    pass
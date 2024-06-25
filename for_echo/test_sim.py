from sklearn.model_selection import train_test_split

from config import parameters as CFG
from kernel import M0_Kernel
from utils import load_file, do_normalisation
import numpy as np
from sklearn import svm
import pandas as pd 
from sklearn.metrics import f1_score, accuracy_score

def do_it():
    normalise_data = CFG["general"]["normalise_data"]

    # set the number of bins
    param_value = None  # use default: log2(num of inst) + 1

    overall_result = {}

    for file_set in CFG["file_names"]:
        result = []
        data_name = file_set["name"]
        print(f"Data set: {data_name}")

        X, Y, data_stats = load_file(file_set)

        if normalise_data:
            X = do_normalisation(X, data_stats)

        # 50% split between train and test
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

                # Calculate F1 score and accuracy
        clf = svm.SVC(kernel='rbf', random_state=42)
        clf.fit(train_X, train_Y)
        # Predict using the precomputed test kernel matrix
        y_pred = clf.predict(test_X)


        f1 = f1_score(test_Y, y_pred,average="macro")
        acc = accuracy_score(test_Y, y_pred)
        result.append(round(f1, 5))
        result.append(round(acc, 5))
        print(f"Before Kernal F1 Score: {f1:.4f}")
        print(f"Before Kernal Accuracy: {acc:.4f}")

        m0_krn = M0_Kernel(None, data_stats)
        m0_krn.set_nbins(param_value)
        train, test = m0_krn.build_model(train_X, test_X)  # this does the pre-processing step

        print("- Sim: Train")
        sim_train = m0_krn.transform(train)

        print("- Sim: Train/Test")
        sim_test = m0_krn.transform(test,train)  # row = train, col = test


        # Calculate F1 score and accuracy
        clf = svm.SVC(kernel='precomputed')
        clf.fit(sim_train, train_Y)
        # Predict using the precomputed test kernel matrix
        y_pred = clf.predict(sim_test)
        f1 = f1_score(test_Y, y_pred,average="macro")
        acc = accuracy_score(test_Y, y_pred)
        result.append(round(f1, 5))
        result.append(round(acc, 5))
        print(f"After Kernal F1 Score: {f1:.4f}")
        print(f"After Kernal Accuracy: {acc:.4f}")
        print()

        overall_result[data_name] = result

    print("All finished!")

    df = pd.DataFrame(overall_result).T
    
    # Rename columns
    df = df.rename(columns={0: "Origin F1", 1: "Origin Acc", 2: "Kernal F1", 3: "Kernal Acc"})

    df.to_csv("compare_result.csv")



if __name__ == '__main__':
    do_it()

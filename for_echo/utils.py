import tomllib

import numpy as np
import pandas as pd

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import minmax_scale, MinMaxScaler


def do_normalisation(data, stats):
    if stats is None:
        scalar = MinMaxScaler()
        scalar.fit(data)
        data = scalar.transform(data)
    else:
        new_X = None
        count = len(stats["attribute"]) - 1

        for i, attr in enumerate(stats["attribute"]):
            if i == count:
                # skip the class attribute
                break

            tmp = data[:, i]

            if "Numeric" in attr["type"]:
                tmp = minmax_scale(tmp)

            if new_X is None:
                new_X = tmp
            else:
                new_X = np.c_[new_X, tmp]

        data = new_X

    return data


def load_file(file_set):

    def load_data():
        if "is_libsvm" in file_set:
            X, Y = load_svmlight_file(file_set["file_name"], dtype = np.float32, order = "C")
            X = X.toarray()
            Y = np.float32(Y)

            nrows, ncols = X.shape
            nclasses = len(np.unique(Y))
        else:
            has_header = file_set["has_header"]
            class_index = file_set["class_index"]

            if has_header:
                hdr = 0
            else:
                hdr = None

            data_file = pd.read_csv(file_set["file_name"], header = hdr)
            nrows = len(data_file.index)
            ncols = len(data_file.columns)

            if class_index == "first":
                class_att_index = 1
            elif class_index == "last":
                class_att_index = ncols
            else:
                class_att_index = int(class_index)

            data_targets = data_file.iloc[:, class_att_index - 1]
            data_feature_values = data_file.drop(data_file.columns[class_att_index - 1], axis = 1)
            X = np.array(data_feature_values, np.float32, order = "C")
            Y = np.array(data_targets, np.float32, order = "C")
            unique_targets = data_targets.unique()
            nclasses = len(unique_targets)
            print("- Class attribute index: " + str(class_att_index))

        # print data statistics
        print("- Number of data records: " + str(nrows))
        print("- Number of attributes: " + str(ncols))
        print("- Number of class labels: " + str(nclasses))

        return X, Y

    def load_stats():
        res = None

        if "file_stats" in file_set:
            with open(file_set["file_stats"], mode = "rb") as f:
                res = tomllib.load(f)
                print("- Found some stats!")
        else:
            print("No stats")
            print(res)
        return res

    X, Y = load_data()
    stats = load_stats()

    return X, Y, stats

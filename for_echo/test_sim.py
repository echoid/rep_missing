from sklearn.model_selection import train_test_split

from config import parameters as CFG
from kernel import M0_Kernel
from utils import load_file, do_normalisation


def do_it():
    normalise_data = CFG["general"]["normalise_data"]

    # set the number of bins
    param_value = None  # use default: log2(num of inst) + 1

    for file_set in CFG["file_names"]:
        data_name = file_set["name"]
        print(f"Data set: {data_name}")

        X, Y, data_stats = load_file(file_set)

        if normalise_data:
            X = do_normalisation(X, data_stats)

        # 50% split between train and test
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.5, random_state=42)

        m0_krn = M0_Kernel(None, data_stats)
        m0_krn.set_nbins(param_value)
        train, test = m0_krn.build_model(train_X, test_X)  # this does the pre-processing step

        print("- Sim: Train")
        sim_train = m0_krn.transform(train)

        print("- Sim: Train/Test")
        sim_test = m0_krn.transform(train, test)  # row = train, col = test

        print()

    print("All finished!")


if __name__ == '__main__':
    do_it()

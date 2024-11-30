"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)  # Number of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mse_loss(e):
    return 1 / 2 * np.mean(e**2)


def mae_loss(e):
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    return mse_loss(e)


def compute_gradient(y, tx, w):
    N = len(y)
    e = y - tx.dot(w)
    gd = -tx.T.dot(e) / N
    return gd


def preprocess_data_train(x_train, X_test):
    columns_to_remove = []

    # Delete columns with more that 50% NaN values
    for i in range(x_train.shape[1]):
        if np.sum(np.isnan(x_train[:, i])) > 0.5 * x_train.shape[0]:
            columns_to_remove.append(i)

    # Change the NaN values to the median of the column
    median_xtrain = np.nanmedian(x_train, axis=0)
    median_xtest = np.nanmedian(X_test, axis=0)
    x_train = np.where(np.isnan(x_train), median_xtrain, x_train)
    X_test = np.where(np.isnan(X_test), median_xtest, X_test)

    # Remove the columns with single unique values
    for i in range(x_train.shape[1]):
        if len(np.unique(x_train[:, i])) == 1:
            if i not in columns_to_remove:
                columns_to_remove.append(i)

    # Remove the columns with high correlation
    corr_matrix = np.corrcoef(x_train, rowvar=False)
    threshold = 0.75
    high_corr = np.where(np.abs(corr_matrix) > threshold)
    high_corr_pairs = [(i, j) for i, j in zip(*high_corr) if i < j]

    for i, j in high_corr_pairs:
        if i not in columns_to_remove and j not in columns_to_remove:
            columns_to_remove.append(j)

    # Remove the Columns
    x_train = np.delete(x_train, columns_to_remove, axis=1)
    X_test = np.delete(X_test, columns_to_remove, axis=1)

    # Remove the outliers (Using Z-Score method)

    # Standardize the data
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train = (x_train - mean) / std
    X_test = (X_test - mean) / std

    # Add a Bias Column
    # bias_x_train = np.ones((x_train.shape[0]))
    # x_train = np.column_stack((bias_x_train, x_train))

    # bias_X_test = np.ones((X_test.shape[0]))
    # X_test = np.column_stack((bias_X_test, X_test))

    return x_train, X_test


def normalize_data(np_array):
    # Normalize the data
    mean = np.mean(np_array, axis=0)
    std = np.std(np_array, axis=0)
    epsilon = 1e-6
    std[std < epsilon] = epsilon
    normalized_data = (np_array - mean) / std
    return normalized_data, mean, std


def split_data(x, y, split_ratio):
    split_index = int(len(x) * split_ratio)
    x_train, x_val = x[:split_index], x[split_index:]

    y_train, y_val = y[:split_index], y[split_index:]

    y_train[np.where(y_train == -1)] = 0
    y_val[np.where(y_val == -1)] = 0

    return x_train, y_train, x_val, y_val


def predict_labels(w, x_test):
    y_pred = np.dot(x_test, w)
    # Print the top 10 w
    print(w[:10])
    # Print the top 10 predictions
    print(y_pred[:10])
    exit()


def check_accuracy(pred, y_test):
    return np.mean(pred == y_test)


def f1_score(pred, y_test):
    tp = np.sum((pred == 1) & (y_test == 1))
    fp = np.sum((pred == 0) & (y_test == 1))
    fn = np.sum((pred == 1) & (y_test == 0))

    f1 = (2 * tp) / (2 * tp + fp + fn)

    print("TP", tp, "FP", fp, "FN", fn)

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return f1, prec, rec

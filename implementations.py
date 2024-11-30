import numpy as np
from helpers import *
import csv
import os


# STEP 2: Implement ML Methods
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)

        w = w - gamma * grad

        loss = compute_loss(y, tx, w)
        losses.append(loss)

        ws.append(w)

    return ws[-1], losses[-1]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
            losses.append(loss)
            ws.append(w)
    return ws[-1], losses[-1]


def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    N = len(y)
    D = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    print("THIS IS MAX ITER", max_iters)
    for n_iter in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w)
        w = w - gamma * grad
        print("THIS IS LOSS")
        loss = compute_logistic_loss(y, tx, w)

        print(loss)
        losses.append(loss)
        ws.append(w)
    if max_iters == 0:
        print("w", ws, "loss", losses)
        return ws, losses
    return ws[-1], losses[-1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_class_weights(y):
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights


def compute_logistic_loss(y, tx, w):
    p = sigmoid(np.dot(tx, w))
    loss = np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
    return loss


def compute_logistic_gradient(y, tx, w):
    p = sigmoid(np.dot(tx, w))
    grad = np.dot(tx.T, p - y) / len(y)
    return grad


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    loss = compute_logistic_loss(y, tx, w)
    for i in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
        loss = compute_logistic_loss(y, tx, w)
        print("Iteration", i, "Loss", loss)
    return w, loss


# Step 3: Generating Good Predictions on the Medical Dataset
def k_fold_split(x, y, k):
    fold_size = len(x) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(x)
        x_fold = x[start:end]
        y_fold = y[start:end]
        folds.append((x_fold, y_fold))
    return folds


def stratified_k_fold(X, y, k):
    # Ensure the data is shuffled
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Sort by class
    sorted_indices = np.argsort(y)
    X = X[sorted_indices]
    y = y[sorted_indices]

    # Split into k folds, ensuring that class distributions are preserved
    folds = []
    fold_sizes = np.full(k, len(y) // k, dtype=int)
    fold_sizes[: len(y) % k] += 1  # Distribute the remainder

    current = 0
    for fold_size in fold_sizes:
        x_fold = X[current : current + fold_size]
        y_fold = y[current : current + fold_size]
        folds.append((x_fold, y_fold))
        current += fold_size

    return folds


def cross_validate(x, y, k=5):  # 5
    folds = stratified_k_fold(x, y, k)
    print("Length of folds", len(folds[1][0]))
    f1_scores = []
    best_w = None
    best_f1_score = -1

    for i in range(k):
        x_val, y_val = folds[i]
        x_train = np.vstack([folds[j][0] for j in range(k) if j != i])
        y_train = np.hstack([folds[j][1] for j in range(k) if j != i])

        w = np.zeros(x_train.shape[1])
        max_iters = 100  # 200
        gamma = 0.01  # 0.8
        lambda_ = 0.001  # 0.00001

        w, loss = reg_logistic_regression(
            y_train, x_train, lambda_, w, max_iters, gamma
        )
        # w = adam_optimizer(
        #    y_train,
        #    x_train,
        #    w,
        #    lambda_,
        #    alpha=0.001,
        #    beta1=0.5,
        #    beta2=0.7,
        #    epsilon=1e-4,
        #    num_iters=200,
        # )
        pred = np.dot(x_val, w)
        pred = np.where(pred >= 0.5, 1, 0)

        f1_score_value, _, _ = f1_score(pred, y_val)
        f1_scores.append(f1_score_value)

        if f1_score_value > best_f1_score:
            best_f1_score = f1_score_value
            best_w = w

    return np.mean(f1_scores), best_w


def oversample_minority_class(X, y):
    # Identify the minority class
    unique, counts = np.unique(y, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]

    # Separate the minority and majority class samples
    minority_samples = X[y == minority_class]
    majority_samples = X[y == majority_class]

    # Calculate the number of samples needed to balance the classes
    num_minority_samples_needed = len(majority_samples) - len(minority_samples)

    # Randomly sample with replacement from the minority class
    indices = np.random.choice(
        len(minority_samples), num_minority_samples_needed, replace=True
    )
    oversampled_minority_samples = minority_samples[indices]

    # Append these samples to the original dataset
    X_balanced = np.vstack((X, oversampled_minority_samples))
    y_balanced = np.hstack((y, np.full(num_minority_samples_needed, minority_class)))

    return X_balanced, y_balanced


def undersample_majority_class(X, y):
    # Identify the majority and minority classes
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    minority_class = unique[np.argmin(counts)]

    # Separate the majority and minority class samples
    majority_samples = X[y == majority_class]
    minority_samples = X[y == minority_class]

    # Calculate the number of samples needed to balance the classes
    num_majority_samples_needed = len(minority_samples)

    # Randomly sample without replacement from the majority class
    indices = np.random.choice(
        len(majority_samples), num_majority_samples_needed, replace=False
    )
    undersampled_majority_samples = majority_samples[indices]

    # Combine the undersampled majority class with the minority class
    X_balanced = np.vstack((undersampled_majority_samples, minority_samples))
    y_balanced = np.hstack(
        (
            np.full(num_majority_samples_needed, majority_class),
            np.full(len(minority_samples), minority_class),
        )
    )

    return X_balanced, y_balanced


def adam_optimizer(
    y, tx, w, lambda_, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iters=1000
):
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    t = 0

    for i in range(num_iters):
        # Compute the gradient
        g = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w

        # Update timestep
        t += 1

        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g

        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g**2)

        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1**t)

        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2**t)

        # Update weights
        w = w - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        print("Iteration", i, "Loss", compute_logistic_loss(y, tx, w))

    return w


def xavier_initialization(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))


def main():
    # Best F1 Score 0.4090474673462886
    if os.path.exists("preprocessed_data.npz"):
        data = np.load("preprocessed_data.npz")
        x_train = data["x_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        train_ids = data["train_ids"]
        test_ids = data["test_ids"]
    else:
        print("Preprocessing data")
        x_train, X_test, y_train, train_ids, test_ids = load_csv_data("./data")
        print("Shape of y_train", y_train.shape)
        print("X_train shape", x_train.shape)
        print("X_test shape", X_test.shape)
        print()
        x_train, X_test = preprocess_data_train(x_train, X_test)

        # Change the y_train -1 to 0
        for i in range(len(y_train)):
            if y_train[i] == -1:
                y_train[i] = 0

        np.savez(
            "preprocessed_data.npz",
            x_train=x_train,
            X_test=X_test,
            y_train=y_train,
            train_ids=train_ids,
            test_ids=test_ids,
        )

    print("X_train shape", x_train.shape)
    print("X_test shape", X_test.shape)

    split_ratio = 0.8
    x_train, y_train, x_test, y_test = split_data(x_train, y_train, split_ratio)
    np.savez(
        "split_data.npz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    data = np.load("split_data.npz")
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    # x_train, y_train = oversample_minority_class(x_train, y_train)
    # x_train, y_train = undersample_majority_class(x_train, y_train)

    print("X_train shape", x_train.shape)
    print("X_test shape", x_test.shape)
    print("Y_train shape", y_train.shape)
    print("Y_test shape", y_test.shape)
    print()

    n_in = x_train.shape[1]
    n_out = 1

    w = xavier_initialization(n_in, n_out)
    f1_score_value, w = cross_validate(x_train, y_train)

    # w, loss = reg_logistic_regression(y_train, x_train, lambda_, w, max_iters, gamma)

    pred = np.dot(x_test, w)
    pred = np.where(pred >= 0.5, 1, 0)

    print("Length of pred", len(pred), "Length of y_test", len(y_test))
    accuracy = check_accuracy(pred, y_test)
    print("Accuracy", accuracy)
    f1_score_value, prec, rec = f1_score(pred, y_test)
    print("F1 Score", f1_score_value, "Precision", prec, "Recall", rec)

    actual_pred = np.dot(X_test, w)
    actual_pred = np.where(actual_pred >= 0.5, 1, -1)

    # How many rows are there

    create_csv_submission(test_ids, actual_pred, "submission.csv")


main()

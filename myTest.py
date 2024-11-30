import numpy as np
import implementations as student_implementations

from conftest import ATOL, RTOL

MAX_ITERS = 2
GAMMA = 0.1


def test_mean_squared_error_gd_0_step(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    expected_w = np.array([0.413044, 0.875757])
    w, loss = student_implementations.mean_squared_error_gd(y, tx, expected_w, 0, GAMMA)

    expected_w = np.array([0.413044, 0.875757])
    expected_loss = 2.959836

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_mean_squared_error_gd(student_implementations, y, tx, initial_w):
    global MAX_ITERS, GAMMA
    w, loss = student_implementations.mean_squared_error_gd(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )

    expected_w = np.array([-0.050586, 0.203718])
    expected_loss = 0.051534

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_mean_squared_error_sgd(student_implementations, y, tx, initial_w):
    global MAX_ITERS, GAMMA
    # n=1 to avoid stochasticity
    w, loss = student_implementations.mean_squared_error_sgd(
        y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.844595
    expected_w = np.array([0.063058, 0.39208])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_least_squares(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    w, loss = student_implementations.least_squares(y, tx)

    expected_w = np.array([0.218786, -0.053837])
    expected_loss = 0.026942

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda0(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    lambda_ = 0.0
    w, loss = student_implementations.ridge_regression(y, tx, lambda_)

    expected_loss = 0.026942
    expected_w = np.array([0.218786, -0.053837])

    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_ridge_regression_lambda1(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    lambda_ = 1.0
    w, loss = student_implementations.ridge_regression(y, tx, lambda_)

    expected_loss = 0.03175
    expected_w = np.array([0.054303, 0.042713])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression_0_step(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    expected_w = np.array([0.463156, 0.939874])
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.logistic_regression(y, tx, expected_w, 0, GAMMA)

    expected_loss = 1.533694

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_logistic_regression(student_implementations, y, tx, initial_w):
    global MAX_ITERS, GAMMA
    y = (y > 0.2) * 1.0
    print("THIS IS W", MAX_ITERS)
    w, loss = student_implementations.logistic_regression(
        y, tx, initial_w, MAX_ITERS, GAMMA
    )
    
    expected_loss = 1.348358
    expected_w = np.array([0.378561, 0.801131])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression(student_implementations, y, tx, initial_w):
    global MAX_ITERS, GAMMA
    lambda_ = 1.0
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.reg_logistic_regression(
        y, tx, lambda_, initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.972165
    expected_w = np.array([0.216062, 0.467747])

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape


def test_reg_logistic_regression_0_step(student_implementations, y, tx):
    global MAX_ITERS, GAMMA
    lambda_ = 1.0
    expected_w = np.array([0.409111, 0.843996])
    y = (y > 0.2) * 1.0
    w, loss = student_implementations.reg_logistic_regression(
        y, tx, lambda_, expected_w, 0, GAMMA
    )

    expected_loss = 1.407327

    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    assert loss.ndim == 0
    assert w.shape == expected_w.shape
    
    
tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
print(tx.shape)
y = np.array([0.1, 0.3, 0.5])
initial_w = np.array([0.5, 1.0])


#test_mean_squared_error_gd_0_step(student_implementations, y, tx)

test_mean_squared_error_gd(student_implementations, y, tx, initial_w)

test_mean_squared_error_sgd(student_implementations, y, tx, initial_w)

test_least_squares(student_implementations, y, tx)

test_ridge_regression_lambda1(student_implementations, y, tx)

test_logistic_regression_0_step(student_implementations, y, tx)

test_logistic_regression(student_implementations, y, tx, initial_w)

test_reg_logistic_regression(student_implementations, y, tx, initial_w)

test_reg_logistic_regression_0_step(student_implementations, y, tx)

print("All tests passed!")

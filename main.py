import math
import numpy as np


def ex2d():
    x = np.array([[0],
                  [1]])
    mew = 0.01
    print("hi")


def objective(x, mew):
    x1 = x[0][0]
    x2 = x[1][0]
    function = math.pow(x1+x2, 2) - 10*(x1+x2)
    penalty1 = mew * math.pow(3*x1 + x2 - 6, 2)
    penalty2 = mew * math.pow(max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5), 2)
    penalty3 = mew * math.pow(max(0.0, -x1), 2)
    objective_x = function + penalty1 + penalty2 + penalty3
    return objective_x


# need to edit
def gradient(x, mew):
    x1 = x[0][0]
    x2 = x[0][1]
    sub_derivative_x1 = 2*x1 + 2*x2 - 10
    penalty1_x1 = mew * math.pow(3, 2)
    penalty2_x1 = mew * (2*x1) * math.pow(max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5), 2)
    penalty3_x1 = mew * (-1) * math.pow(max(0.0, -x1), 2)
    derivative_x1 = sub_derivative_x1 + penalty1_x1 + penalty2_x1 + penalty3_x1

    sub_derivative_x2 = 2 * x1 + 2 * x2 - 10
    penalty1_x2 = mew
    penalty2_x2 = mew * (2*x2) * math.pow(max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5), 2)
    penalty3_x2 = 0
    derivative_x2 = sub_derivative_x2 + penalty1_x2 + penalty2_x2 + penalty3_x2

    gradient_x = np.array([[derivative_x1],
                           [derivative_x2]])
    return gradient_x


'''
def Steepest_Descent(w, x_train, y_train, x_test, y_test, alpha=1.0, iterations=100):
    train_cost_history = []
    test_cost_history = []
    d = np.zeros(w.shape)
    g_k = np.zeros(w.shape)
    f_k = 1
    for i in range(iterations):
        w = np.clip(w, -1, 1)
        if i != 0:
            alpha = Armijo_Linesearch(w, x_train, y_train, d, g_k)
        w += alpha * d
        if f_k < 1e-3:
            break
        f_k, g_k = Logistic_Regression(w, x_train, y_train, hessian_indicator=False)
        d = -np.array(g_k)
        f_0 = cost(w, x_test, y_test)
        train_cost_history.append(f_k[0][0])
        test_cost_history.append(f_0[0][0])
    return w, np.array(train_cost_history), np.array(test_cost_history)


def Armijo_Linesearch(w, x, y, d, g_k, alpha=1.0, beta=0.8, c=1e-5):
    f_k = cost(w, x, y)
    for i in range(10):
        f_k_1 = cost(w + (alpha * d), x, y)
        if f_k_1 <= f_k + (alpha * c * np.dot(d.transpose(), g_k)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha
'''

if __name__ == '__main__':
    ex2d()


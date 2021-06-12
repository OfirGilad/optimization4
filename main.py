import math
import numpy as np
import matplotlib.pyplot as plt


def ex2d():
    x = np.array([[0.0],
                  [0.0]])
    result_x, objective_history, constraint1_history, constraint2_history, constraint3_history = Steepest_Descent(x)
    fig1, axs1 = plt.subplots()
    fig2, axs2 = plt.subplots()
    fig3, axs3 = plt.subplots()
    fig4, axs4 = plt.subplots()
    axs1.plot(objective_history, label="objective")
    axs2.plot(constraint1_history, label="constraint1")
    axs3.plot(constraint2_history, label="constraint2")
    axs4.plot(constraint3_history, label="constraint3")
    axs1.set_ylabel('Value')
    axs2.set_ylabel('Value')
    axs3.set_ylabel('Value')
    axs4.set_ylabel('Value')
    axs1.set_xlabel('Iterations')
    axs2.set_xlabel('Iterations')
    axs3.set_xlabel('Iterations')
    axs4.set_xlabel('Iterations')
    axs1.set_title('Function values')
    axs2.set_title('Constraint1 violation')
    axs3.set_title('Constraint2 violation')
    axs4.set_title('Constraint3 violation')
    axs1.legend()
    axs2.legend()
    axs3.legend()
    axs4.legend()
    plt.show()
    print("x = ", result_x)


def objective(x, mu):
    x1 = x[0][0]
    x2 = x[1][0]
    function = math.pow(x1+x2, 2) - 10*(x1+x2)
    penalty1 = mu * math.pow(3*x1 + x2 - 6, 2)
    penalty2 = mu * math.pow(max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5), 2)
    penalty3 = mu * math.pow(max(0.0, -x1), 2)
    objective_x = function + penalty1 + penalty2 + penalty3
    return objective_x


def gradient(x, mu):
    x1 = x[0][0]
    x2 = x[1][0]
    sub_derivative_x1 = 2*x1 + 2*x2 - 10
    penalty1_x1 = mu * (18*x1 + 6*x2 - 36)
    penalty2_x1 = mu * max(0.0, 4*math.pow(x1, 3) + 4*x1*math.pow(x2, 2) - 20*x1)
    penalty3_x1 = mu * max(0.0, -2*x1)
    derivative_x1 = sub_derivative_x1 + penalty1_x1 + penalty2_x1 + penalty3_x1

    sub_derivative_x2 = 2 * x1 + 2 * x2 - 10
    penalty1_x2 = mu * (2*x2 + 6*x1 - 12)
    penalty2_x2 = mu * max(0.0, 4*math.pow(x2, 3) + 4*x2*math.pow(x1, 2) - 20*x2)
    penalty3_x2 = 0
    derivative_x2 = sub_derivative_x2 + penalty1_x2 + penalty2_x2 + penalty3_x2

    gradient_x = np.array([[derivative_x1],
                           [derivative_x2]])
    return gradient_x


def Steepest_Descent(x, alpha=1.0, iterations=50):
    mu_vector = [0.01, 0.1, 1.0, 10.0, 100.0]
    objective_history = []
    constraint1_history = []
    constraint2_history = []
    constraint3_history = []
    current_x = x
    for mu in mu_vector:
        print("mu = " + str(mu))
        for i in range(iterations):
            gradient_x = gradient(current_x, mu)
            if np.linalg.norm(gradient_x) < 1e-3:
                break
            d = -gradient_x
            alpha = Armijo_Linesearch(mu, current_x, d, gradient_x, alpha=alpha)
            current_x += alpha * d
            objective_x = objective(current_x, mu)
            objective_history.append(objective_x)
            x1 = x[0][0]
            x2 = x[1][0]
            constraint1_history.append(np.linalg.norm(3 * x1 + x2 - 6))
            constraint2_history.append(np.linalg.norm((max(0.0, math.pow(x1, 2) + math.pow(x2, 2) - 5))))
            constraint3_history.append(np.linalg.norm((max(0.0, -x1))))

    return current_x, objective_history, constraint1_history, constraint2_history, constraint3_history


def Armijo_Linesearch(mu, x, d, gradient_x, alpha=1.0, beta=0.5, c=1e-5):
    objective_x = objective(x, mu)
    for i in range(10):
        objective_x_1 = objective(x + (alpha * d), mu)
        if objective_x_1 <= objective_x + (alpha * c * np.dot(d.transpose(), gradient_x)):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def ex3cd():
    H = np.array([[5, -1, -1, -1, -1],
                  [-1, 5, -1, -1, -1],
                  [-1, -1, 5, -1, -1],
                  [-1, -1, -1, 5, -1],
                  [-1, -1, -1, -1, 5]])
    g = np.array([[18],
                  [6],
                  [-12],
                  [-6],
                  [18]])
    a = np.array([[0],
                  [0],
                  [0],
                  [0],
                  [0]])
    b = np.array([[5],
                  [5],
                  [5],
                  [5],
                  [5]])
    result_x = Coordinate_Descent(H, g, a, b)


def Coordinate_Descent(H, g, a, b, iterations=100):
    n = a.shape[0]
    current_x = np.zeros((n, 1))
    objective_history = []
    for i in range(iterations):
        for j in range(n):
            ################## in progress
            current_x = 0
    return current_x, objective_history


if __name__ == '__main__':
    ex2d()


import numpy as np


def ex3c(H, g, x0, a, b, max_iter=100):
    """
    :param H:matrix
    :param g:vector
    :param x0: initial guess
    :param a: vector , lower bound constraint
    :param b: vector , upper bound constraint
    :return: final x
    """

    H_diag = np.diag(np.diag(H))
    H_rest = H - H_diag
    x = x0
    curr_norm = np.linalg.norm(x)
    for _ in range(max_iter):
        for i in range(len(x)):
            x[i] = calc_xi(H_rest, H[i][i], g, x, a, b, i)
        last_norm = curr_norm
        curr_norm = np.linalg.norm(x)
        if abs(last_norm - curr_norm) < 0.001:
            break
    return x


def calc_xi(H_rest, Hii, g, x, a, b, i):
    """
    :param Hii: the cordinet (i,i) of the H matrix
    :param H:matrix
    :param g:vector
    :param x0: initial guess
    :param a: vector , lower bound constraint
    :param b: vector , upper bound constraint
    :return: final x
    """
    xi = - (H_rest[i] @ x - g[i])
    xi /= Hii
    # return xi
    return apply_contrains(xi, a, b, i)


def apply_contrains(xi, a, b, i):
    xi = min(xi, b[i])
    xi = max(xi, a[i])
    return xi


def init_h(n):
    H = np.ones(n)
    H *= -1
    return H + np.diag([6] * n)


if __name__ == "__main__":
    n = 5
    H = init_h(n)
    g = np.array([18, 6, -12, -6, 18])
    a = np.array([0] * n)
    b = np.array([5] * n)
    x0 = np.array([1] * n, dtype=np.float_)
    ans = ex3c(H, g, x0, a, b)
    print(ans)

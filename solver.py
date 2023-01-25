import scipy.integrate as sp
import numpy as np
import matplotlib.pyplot as plot


def E(x):
    return 2.0 if 0.0 <= x <= 1.0 else 6.0


# returns value of B(e_i, e_j)
def B(w, v, der_w, der_v, lower, upper):
    f1 = lambda x: 2.0 * der_w(x) * der_v(x)
    f2 = lambda x: 6.0 * der_w(x) * der_v(x)

    i1 = sp.quad(f1, 0, 1, epsabs=10e-10, limit=100)
    i2 = sp.quad(f2, 1, 2, epsabs=10e-10, limit=100)
    return i1[0] + i2[0] - 2.0 * w(0) * v(0)


# returns value of L(v) in this case it's constant
def L(v):
    return (-1) * 20.0 * v(0)


def u_():
    return lambda x: 3.0


def u_prim():
    return lambda x: 0.0


def make_matrix(n, left, right):
    b_matrix = np.zeros((n + 1, n + 1))
    l_matrix = np.zeros(n + 1)
    h = 2 / n
    for i in range(n):
        for j in range(n + 1):

            if abs(i - j) > 1:
                b_matrix[i][j] = 0.0

            else:

                b_matrix[i][j] = B(e_i(j, n), e_i(i, n),
                                   de_i(j, n), de_i(i, n),
                                   left, right)
        l_matrix[i] = - B(u_(), e_i(i, n), u_prim(), de_i(i, n), left,
                          right) + L(e_i(i, n))
    for i in range(n):
        b_matrix[n][i] = 0.0
    b_matrix[n][n] = 1.0
    l_matrix[n] = 3.0

    return b_matrix, l_matrix


def draw_chart(wsp, n):
    x = np.arange(0.0, 2.0, 0.0001)
    y = np.zeros(20000)

    for i in range(20000):
        for j in range(n + 1):
            y[i] += wsp[j] * e_i(j, n)(x[i])

    plot.plot(x, y)
    plot.show()


# ======================================================= nowe =====================
def e_i(i, n):
    h = 2 / n
    return lambda x: (x - (i - 1) * h) / (i * h - (i - 1) * h) if (i - 1) * h <= x <= i * h else (h * (i + 1) - x) / (
            h * (i + 1) - h * i) if (i * h < x < (i + 1) * h) else 0.0


def de_i(i, n):
    h = 2 / n
    return lambda x: 1 / (i * h - (i - 1) * h) if (i - 1) * h <= x <= i * h else -1 / (
        (((i + 1) * h) - (i * h))) if (i * h < x < (i + 1) * h) else 0.0


# ======================================= testy
if __name__ == "__main__":
    # n = int(input("n = "))
    # l = int(input("l = "))
    # r = int(input("r = "))

    n = 25
    matrixes = make_matrix(n, 0, 2)
    b_u_v = matrixes[0]
    l_v = matrixes[1]

    print(b_u_v)
    print(l_v.tolist())

    ans = np.linalg.solve(b_u_v, l_v)

    draw_chart(ans, n)

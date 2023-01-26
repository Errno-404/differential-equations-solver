# Adam Mężydło - MES


import scipy.integrate as sp
import numpy as np
import matplotlib.pyplot as plot


def E(x):
    return 2.0 if 0.0 <= x <= 1 else 6.0


# zwraca obliczoną wartość B(ei, ej)
def B(w, v, dw, dv, left, right):
    integral = sp.quad(lambda x: E(x) * dw(x) * dv(x), left, right, limit=100)
    return integral[0] - 2.0 * w(0) * v(0)


# zwraca wartość L(v) przed shiftem
def L(v):
    return (-1) * 20.0 * v(0)


# funkcja shift (u z daszkiem)
def u_():
    return lambda x: 3.0


# pochodna funkcji (u z daszkiem)
def du_():
    return lambda x: 0.0


# funkcja tworząca macierze B(w, v) oraz L(v)
def make_matrix(n):
    b_matrix = np.zeros((n + 1, n + 1))
    l_matrix = np.zeros(n + 1)
    h = 2 / n

    for i in range(n):
        for j in range(n + 1):

            if abs(i - j) > 1:
                b_matrix[i][j] = 0.0

            else:
                left = max((min(i, j) - 1) * h, 0.0)
                right = min((max(i, j) + 1) * h, 2.0)

                # print("i: ", i, "j: ", j, "left: ", left, "right: ", right)

                b_matrix[i][j] = B(e_i(j, n), e_i(i, n), de_i(j, n), de_i(i, n), left, right)

    for i in range(n):
        left = max((i - 1) * h, 0.0)
        right = min((i + 1) * h, 2.0)
        l_matrix[i] = - B(u_(), e_i(i, n), du_(), de_i(i, n),left, right) + L(e_i(i, n))

    for i in range(n):
        b_matrix[n][i] = 0.0
    b_matrix[n][n] = 1.0
    l_matrix[n] = 3.0

    return b_matrix, l_matrix


# funkcja rysująca wykres
def draw_chart(wsp, n):
    x = np.arange(0.0, 2.0, 0.0001)
    y = np.zeros(20000)

    for i in range(20000):
        for j in range(n + 1):
            y[i] += wsp[j] * e_i(j, n)(x[i])

    plot.plot(x, y)
    plot.show()


# Funkcja testująca
def e_i(i, n):
    h = 2 / n
    return lambda x: (x - (i - 1) * h) / (i * h - (i - 1) * h) if (i - 1) * h <= x <= i * h else (h * (i + 1) - x) / (
            h * (i + 1) - h * i) if (i * h < x < (i + 1) * h) else 0.0


# Pochodna funkcji testującej
def de_i(i, n):
    h = 2 / n
    return lambda x: 1 / (i * h - (i - 1) * h) if (i - 1) * h <= x <= i * h else -1 / (
        (((i + 1) * h) - (i * h))) if (i * h < x < (i + 1) * h) else 0.0


if __name__ == "__main__":
    N = int(input("n = "))

    matrixes = make_matrix(N)
    b_u_v = matrixes[0]
    l_v = matrixes[1]

    # print(b_u_v)
    # print(l_v.tolist())

    ans = np.linalg.solve(b_u_v, l_v)

    draw_chart(ans, N)

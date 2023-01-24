import scipy.integrate as sp
import numpy as np
import matplotlib.pyplot as plot


def E(x):
    return 2 if 0 <= x <= 1 else 6


# returns value of B(e_i, e_j)
def B(w, v, der_w, der_v, lower, upper):
    integral = sp.quad(lambda x: E(x) * der_w * der_v, lower, upper)
    return - 2 * w(0) * v(0) + integral[0]


# returns value of L(v) in this case it's constant
def L(v):
    return -20 * v(0)


def draw_plot(n):
    fux = np.arange(0.0, 1.0, 0.001)
    fuy = np.zeros(1000)
    for i in range(0, 1000):
        for j in range(0, n):
            fuy[i] += e(j, n)(fux[i])

    plot.plot(fux, fuy)
    plot.show()


# funkcje testujace
def e(i, n, left, right):
    delta = (right - left) / n
    return lambda x: 1.0 - abs(x / delta - i) if 1.0 - abs(x / delta - i) >= 0 else 0.0

# pochodne funkcji testujących
def e_prim(i, n, left, right):
    delta = (right - left) / n

    return lambda x: delta if (i - 1) * delta < x < i * delta else -1 * delta if i * delta < x < (
                i + 1) * delta else 0.0


if __name__ == "__main__":


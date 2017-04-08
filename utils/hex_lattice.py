# hex_lattice.py

import numpy as np



def gen_hex_lattice(w, a=1.):
    """
    Fill a circle centered at the origin with radius w
        with a hexagonal lattice nodes separted by a
    """
    L = 2 * w / a + 1
    i, j = np.meshgrid(np.arange(-L, L), np.arange(-L, L))
    i, j = i.ravel()[None, :], j.ravel()[None, :]
    # Random rotation
    theta = np.random.rand(1)[0] * 2 * np.pi
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    u, v = np.array([[1], [0]]), np.array([[0.5], [np.sqrt(3)/2]])
    u, v = np.dot(R, u), np.dot(R, v)
    XE, YE = a * (u * i + v * j)
    idx = XE ** 2 + YE ** 2 < w ** 2
    XE, YE = XE[idx], YE[idx]
    # Random center
    i0, j0 = (np.random.rand(2) - 0.5)
    XE, YE = XE + i0 * a, YE + i0 * a
    return XE, YE


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for i in range(10):
        XE, YE = gen_hex_lattice(5.01, 0.5)
        print len(XE)


    plt.scatter(XE, YE)
    plt.axes().set_aspect('equal')
    plt.show()

#    idx = (-w < XE) * (XE < w) * (-w < YE) * (YE < w)

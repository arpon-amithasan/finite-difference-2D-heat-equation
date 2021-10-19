import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import linalg
from time import process_time


def is_int(x):
    if x == int(x):
        return 1
    else:
        return 0


def is_on_edge(k, nx, total):
    if k < nx:                          # left
        return 1
    elif is_int(k / nx) == 1:           # top
        return 2
    elif is_int((k + 1) / nx) == 1:     # bottom
        return 3
    elif k + nx > total:                # right
        return 4
    else:
        return 0                        # not on edge


def main():

    nx = int(input("Number of points along x = "))
    ny = int(input("Number of points along y = "))
    total = nx * ny

    A = sp.lil_matrix((total, total), dtype=float)           # making A matrix

    A.setdiag(1, -nx)
    A.setdiag(1, -1)
    A.setdiag(-4, 0)
    A.setdiag(1, 1)
    A.setdiag(1, nx)

    for k in range(0, total):
        if is_on_edge(k, nx, total) > 0:
            A[k, :] = 0
            A[k, k] = 1
        else:
            pass

    b = np.zeros((total, 1), dtype=float)       # making b matrix

    b_left = float(input("Temperature at left edge = "))
    b_top = float(input("Temperature at top edge = "))
    b_bottom = float(input("Temperature at bottom edge = "))
    b_right = float(input("Temperature at right edge = "))

    for k in range(0, total):

        if is_on_edge(k, nx, total) == 1:       # left
            b[k, 0] = b_left
        elif is_on_edge(k, nx, total) == 2:     # top
            b[k, 0] = b_top
        elif is_on_edge(k, nx, total) == 3:     # bottom
            b[k, 0] = b_bottom
        elif is_on_edge(k, nx, total) == 4:     # right
            b[k, 0] = b_right
        else:
            pass
    b[0, 0] = float(input("Temperature at top-left corner = "))                            # top-left
    b[nx - 1, 0] = float(input("Temperature at bottom-left corner = "))                       # bottom-left
    b[total - nx, 0] = float(input("Temperature at top-right corner = "))                   # top-right
    b[total - 1, 0] = float(input("Temperature at bottom-right = "))                    # bottom-right

    A = A.tocsc()

    start_time = process_time()
    y, exit_code = sp.linalg.gmres(A, b)
    end_time = process_time()

    print(end_time - start_time, "seconds")

    print(exit_code)

    solution = np.reshape(y, (nx, ny), order='F')

    plt.figure(figsize=(7, 7))
    plot = plt.imshow(solution)  # Plotting the system
    plot.set_cmap("plasma")
    plt.xlabel("440 cm")
    plt.ylabel("440 cm")
    cbar = plt.colorbar()
    cbar.set_label("temperature gradient")
    plt.show()


main()

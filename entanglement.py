import numpy as np

if __name__ == '__main__':
    phi = np.array([[1, 0, 0, 1], [1, 0, 0, 1]]) / np.sqrt(4)
    u, gamma, v = np.linalg.svd(phi, full_matrices=False)
    a = u @ np.diag(gamma) @ v
    pass

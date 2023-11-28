import numpy as np

N = 3

if __name__ == '__main__':
    a = np.arange(0, N**2)
    a = a.reshape((N, N))
    b = np.arange(N**2, 2*N**2)
    b = a.reshape((N, N))
    c = np.arange(2*N**2, 3*N**2)
    c = a.reshape((N, N))
    d = np.arange(3*N**2, 4*N**2)
    d = a.reshape((N, N))
    e = np.arange(4*N**2, 5*N**2)
    e = a.reshape((N, N))

    print(np.trace(a@b@c@d@e))
    print(np.einsum('ij,jk,kl,lm,mi', a, b, c, d, e))
    
    tr = 0
    cnt = 0
    for i, j, k, l, m in np.ndindex(N, N, N, N, N):
        tr += a[i, j] * b[j, k] * c[k, l] * d[l, m] * e[m, i]
        cnt += 1

    print(tr, cnt)

    cnt = 0
    ab = np.zeros((N, N))
    for i, j in np.ndindex(N, N):
        ab[i, j] = a[i, :] @ b[:, j]
        cnt += N

    abc = np.zeros((N, N))
    for i, k in np.ndindex(N, N):
        abc[i, k] = ab[i, :] @ c[:, k]
        cnt += N
    
    abcd = np.zeros((N, N))
    for i, l in np.ndindex(N, N):
        abcd[i, l] = abc[i, :] @ d[:, l]
        cnt += N

    abcde = np.zeros((N, N))
    for i, m in np.ndindex(N, N):
        abcde[i, m] = abcd[i, :] @ e[:, m]
        cnt += N

    tr = 0
    for i in range(N):
        tr += abcde[i, i]
        cnt += 1

    print(tr, cnt)
import numpy as np

if __name__ == '__main__':
    p = np.arange(0, 8)
    p = p.reshape((2, 2, 2))
    q = np.arange(8, 16)
    q = p.reshape((2, 2, 2))
    r = np.arange(16, 24)
    r = p.reshape((2, 2, 2))

    print(np.einsum('ije,ekf,flm,ija,akb,blm', p, q, r, p, q, r))

    cnt = 0
    pp = np.zeros((2, 2))
    for e, a in np.ndindex(2, 2):
        pp[e, a] = np.sum(p[:, :, e] * p[:, :, a])
        cnt += 4

    pqp = np.zeros((2, 2, 2))
    for f, k, a in np.ndindex(2, 2, 2):
        pqp[a, k, f] = np.sum(pp[:, a] * q[:, k, f])
        cnt += 2

        pqpq = np.zeros((2, 2))
    for f, b in np.ndindex(2, 2):
        pqpq[f, b] = np.sum(pqp[:, :, f] * q[:, :, b])
        cnt += 4

    pqrpq = np.zeros((2, 2, 2))
    for b, l, m in np.ndindex(2, 2, 2):
        pqrpq[b, l, m] = np.sum(pqpq[:, b] * r[:, l, m])
        cnt += 2

    pqrpqr = np.sum(pqrpq * r)
    cnt += 8

    print(pqrpqr, cnt)
    
    
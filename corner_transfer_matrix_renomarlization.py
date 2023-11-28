import numpy as np

CHI = 1024
N = 14

if __name__ == '__main__':
    w = np.zeros((2, 2, 2, 2))
    w[0,0,0,1] = w[0,0,1,0] = w[0,1,0,0] = w[1,0,0,0] = 1
    cij = w[0,:,:,0]
    qjmg = pjmg = w[:,:,:,0]

    for i in range(1, N):
        u, d, v = np.linalg.svd(cij, full_matrices=True)
        print(np.sum(d**4))
        chi = min(len(d), CHI)
        d = np.diag(d[:chi])
        u_bar = u[:,:chi]
        v_bar = v.T[:,:chi]
        p_bar = np.tensordot(u_bar, np.tensordot(pjmg, u_bar, axes=(2, 0)), axes=(0, 0))
        q_bar = np.tensordot(np.tensordot(v_bar, qjmg, axes=(0, 0)), v_bar, axes=(2, 0))
        cij = np.tensordot(p_bar, w, axes=(1, 0))
        cij = np.tensordot(cij, d, axes=(1, 0))
        cij = np.tensordot(cij, q_bar, axes=([4, 3], [0, 1]))
        cij = cij.transpose(1, 0, 2, 3)
        cij = cij.reshape(chi*2, chi*2)
        pjmg = np.tensordot(p_bar, w, axes=(1, 0)).transpose(2, 0, 3, 4, 1)
        qjmg = np.tensordot(q_bar, w, axes=(1, 3)).transpose(2, 0, 3, 4, 1)
        pjmg = pjmg.reshape(chi*2, 2, chi*2)
        qjmg = qjmg.reshape(chi*2, 2, chi*2)



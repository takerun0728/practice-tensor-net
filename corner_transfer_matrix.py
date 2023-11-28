import numpy as np

N = 14

if __name__ == '__main__':
    w = np.zeros((2, 2, 2, 2))
    w[0,0,0,1] = w[0,0,1,0] = w[0,1,0,0] = w[1,0,0,0] = 1
    cij = w[0,:,:,0]
    pjmg = w[:,:,:,0]
    print(np.trace(cij@cij@cij@cij))

    for n in range(1,N):
        c_bnj = np.tensordot(pjmg, cij, axes=(2, 0))
        c_bnmg = np.tensordot(c_bnj, pjmg, axes=(2, 0))
        c_bgaf = np.tensordot(c_bnmg, w, axes=([1,2], [0,3]))
        c_abfg = c_bgaf.transpose(2, 0, 3, 1)
        cij = c_abfg.reshape(2**(n+1), 2**(n+1))

        pgyfdx = np.tensordot(pjmg, w, axes=(1, 3))
        pfgdxy = pgyfdx.transpose(2, 0, 3, 4, 1)
        pjmg = pfgdxy.reshape(2**(n+1), 2, 2**(n+1))

        print(np.trace(cij@cij@cij@cij))

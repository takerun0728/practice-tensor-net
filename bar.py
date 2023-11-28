import numpy as np

def calc_loop_combination(n, m):
    w = np.zeros((4, 4, 4, 4))
    w[0,0,1,0] = w[1,0,2,0] = w[2,0,3,0] = w[3,0,0,0] = w[0,0,0,1] = w[0,1,0,2] = w[0,2,0,3] = w[0,3,0,0] = 1# = w[0,0,0,0] = 1

    mp1 = w[0,:,:,0]
    mp2 = w[0,0,:,:]
    trans = w[0,:,:,:]

    for i in range(n-2):
        mp1 = np.tensordot(mp1, w[:,:,:,0], axes=(i+1, 0))
        mp2 = np.tensordot(mp2, w[:,0,:,:], axes=(i, 0))
        trans = np.tensordot(trans, w, axes=(2*i+1, 0))
    
    mp1 = np.tensordot(mp1, w[:,:,0,0], axes=(n-1, 0))
    mp2 = np.tensordot(mp2, w[:,0,0,:], axes=(n-2, 0))
    trans = np.tensordot(trans, w[:,:,0,:], axes=(2*n-3, 0))

    for j in range(m-2):
        mp1 = np.tensordot(mp1, trans, axes=(range(n), range(1, 2*n, 2)))
    pass
    return np.tensordot(mp1, mp2, axes=(range(n), range(n)))

if __name__ == '__main__':
    for i in range(2, 13):
        comb = calc_loop_combination(i, 4)
        print(f'{i}:{comb},{np.log(comb)/i**2}')

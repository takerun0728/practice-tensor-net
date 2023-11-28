import numpy as np

def calc_tatami_combination(n, m):
    w = np.zeros((2, 2, 2, 2))
    w[1,0,0,0] = w[0,1,0,0] = w[0,0,1,0] = w[0,0,0,1] = 1

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
    for i in range(2, 13, 2):
        comb = calc_tatami_combination(i, i)
        print(f'{i}:{comb},{np.log(comb)/i**2}')

    for i in range(2, 13, 2):
        comb1 = calc_tatami_combination(i, i)
        comb2 = calc_tatami_combination(i+2, i+2)
        comb3 = calc_tatami_combination(i+2, i)
        comb4 = calc_tatami_combination(i, i+2)
        print((np.log(comb1) + np.log(comb2) - np.log(comb3) - np.log(comb4))/4)
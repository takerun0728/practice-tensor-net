import numpy as np

N = 3
CHI = 25

if __name__ == '__main__':
    np.random.seed(0)

    chi2 = min(CHI, N*N)
    chi3 = min(CHI, chi2*N)
    chi4 = min(CHI, chi3*N)
    T_a_b_c_d_e_f_g_h = np.random.randint(0, 10, (N, N, N, N, N, N, N, N))
    A_ab_xi, d, v = np.linalg.svd(T_a_b_c_d_e_f_g_h.reshape(N*N, N*N*N*N*N*N), full_matrices=False)
    A_ab_xi = A_ab_xi[:,:chi2]
    T_xic_defgh = (np.diag(d[:chi2]) @ v[:chi2,:]).reshape(chi2*N, N*N*N*N*N)
    A_xic_mu, d, v = np.linalg.svd(T_xic_defgh, full_matrices=False)
    A_xic_mu = A_xic_mu[:,:chi3]
    T_mudef_gh = (np.diag(d[:chi3]) @ v[:chi3,:]).reshape(chi3*N*N*N, N*N)
    u, d, B_sig_gh = np.linalg.svd(T_mudef_gh, full_matrices=False)
    B_sig_gh = B_sig_gh[:chi2,:]
    T_mude_fsig = (u[:,:CHI] @ np.diag(d[:CHI])).reshape(chi3*N*N, N*chi2)
    u, d, B_rho_fsig = np.linalg.svd(T_mude_fsig, full_matrices=False)
    B_rho_fsig = B_rho_fsig[:chi3,:]
    T_mud_erho = (u[:,:CHI] @ np.diag(d[:CHI])).reshape(chi3*N, N*chi3)
    A_mud_nu, d_nu, B_nu_erho = np.linalg.svd(T_mud_erho, full_matrices=False)
    A_mud_nu = A_mud_nu[:,:chi4]
    B_nu_erho = B_nu_erho[:chi4,:]
    A_a_b_xi = A_ab_xi.reshape(N, N, chi2)
    A_xi_c_mu = A_xic_mu.reshape(chi2, N, chi3)[:chi2,:,:chi3]
    A_mu_d_nu = A_mud_nu.reshape(chi3, N, chi4)
    B_nu_e_rho = B_nu_erho.reshape(chi4, N, chi3)
    B_rho_f_sig = B_rho_fsig.reshape(chi3, N, chi2)
    B_sig_g_h = B_sig_gh.reshape(chi2, N, N)
    A_a_b_c_mu = np.tensordot(A_a_b_xi, A_xi_c_mu, axes=(2, 0))
    A_a_b_c_d_nu = np.tensordot(A_a_b_c_mu, A_mu_d_nu, axes=(3, 0))
    B_rho_f_g_h = np.tensordot(B_rho_f_sig, B_sig_g_h, axes=(2, 0))
    B_nu_e_f_g_h = np.tensordot(B_nu_e_rho, B_rho_f_g_h, axes=(2, 0))
    Tbar_a_b_c_d_e_f_g_h = np.tensordot(np.tensordot(A_a_b_c_d_nu, np.diag(d_nu[:CHI]), axes=(4, 0)), B_nu_e_f_g_h, axes=(4, 0))
    print(A_a_b_xi.size + A_xi_c_mu.size + A_mu_d_nu.size + CHI + B_nu_e_rho.size + B_rho_f_sig.size + B_sig_g_h.size)
    print(N**8)

    print(np.tensordot(T_a_b_c_d_e_f_g_h, T_a_b_c_d_e_f_g_h, axes=([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])))
    print(np.sum(d_nu**2))
    
    pass

import numpy as np

if __name__ == '__main__':
    t_a_b_c_d_e_f_g_h = np.random.randint(4, 10, (2, 2, 2, 2, 2, 2, 2, 2))
    t_ab_cdefgh = t_a_b_c_d_e_f_g_h.reshape(2**2, 2**6)
    uab_xi, d_xi, v_xi_cdefgh = np.linalg.svd(t_ab_cdefgh, full_matrices=False)
    t_xi_cdefgh = np.diag(d_xi) @ v_xi_cdefgh
    t_xi_cd_efgh = t_xi_cdefgh.reshape(2**2, 2**2, 2**4)
    t_cd_xi_efgh = t_xi_cd_efgh.transpose(1, 0, 2)
    a = np.tensordot(uab_xi, t_xi_cd_efgh, axes=(1, 0))  
    t_cd_xiefgh = t_cd_xi_efgh.reshape(2**2, 2**6)
    ucd_mu, d_mu, v_mu_xiefgh =  np.linalg.svd(t_cd_xiefgh, full_matrices=False)
    t_mu_xiefgh = np.diag(d_mu) @ v_mu_xiefgh
    t_mu_xi_ef_gh = t_mu_xiefgh.reshape(2**2, 2**2, 2**2, 2**2)
    t_ef_xi_mu_gh = t_mu_xi_ef_gh.transpose(2, 1, 0, 3)
    t_ef_ximugh = t_ef_xi_mu_gh.reshape(2**2, 2**6)
    uef_nu, d_nu, v_nu_ximugh = np.linalg.svd(t_ef_ximugh, full_matrices=False)
    t_nu_ximugh = np.diag(d_nu) @ v_nu_ximugh
    t_nu_xi_mu_gh = t_nu_ximugh.reshape(2**2, 2**2, 2**2, 2**2)
    t_gh_xi_mu_nu = t_nu_xi_mu_gh.transpose(3, 1, 2, 0)
    t_gh_ximunu = t_gh_xi_mu_nu.reshape(2**2, 2**6)
    ugh_rho, d_rho, v_rho_ximunu = np.linalg.svd(t_gh_ximunu, full_matrices=False)
    t_rho_ximunu = np.diag(d_rho) @ v_rho_ximunu
    t_rho_xi_mu_nu = t_rho_ximunu.reshape(2**2, 2**2, 2**2, 2**2)
    t_xi_mu_nu_rho = t_rho_xi_mu_nu.transpose(1, 2, 3, 0)
    t_ab_mu_nu_rho = np.tensordot(uab_xi, t_xi_mu_nu_rho,  axes=(1, 0))
    t_cd_ab_nu_rho = np.tensordot(ucd_mu ,t_ab_mu_nu_rho, axes=(1, 1))
    t_ef_cd_ab_rho = np.tensordot(uef_nu, t_cd_ab_nu_rho, axes=(1, 2))
    t_gh_ef_cd_ab = np.tensordot(ugh_rho, t_ef_cd_ab_rho, axes=(1, 3))
    t_ab_cd_ef_gh_recov = t_gh_ef_cd_ab.transpose(3, 1, 2, 0)

    #low rank approximation(rank=2)
    t_xi_mu_nu_rho2 = t_xi_mu_nu_rho[:2,:2,:2,:2]
    t_ab_mu_nu_rho2 = np.tensordot(uab_xi[:,:2], t_xi_mu_nu_rho2,  axes=(1, 0))
    t_cd_ab_nu_rho2 = np.tensordot(ucd_mu[:,:2],t_ab_mu_nu_rho2, axes=(1, 1))
    t_ef_cd_ab_rho2 = np.tensordot(uef_nu[:,:2], t_cd_ab_nu_rho2, axes=(1, 2))
    t_gh_ef_cd_ab2 = np.tensordot(ugh_rho[:,:2], t_ef_cd_ab_rho2, axes=(1, 3))
    t_ab_cd_ef_gh2 = t_gh_ef_cd_ab2.transpose(3, 2, 1, 0)

    #low rank approximation(rank=3)
    t_xi_mu_nu_rho3 = t_xi_mu_nu_rho[:3,:4,:4,:4]
    t_ab_mu_nu_rho3 = np.tensordot(uab_xi[:,:3], t_xi_mu_nu_rho3,  axes=(1, 0))
    t_cd_ab_nu_rho3 = np.tensordot(ucd_mu[:,:4],t_ab_mu_nu_rho3, axes=(1, 1))
    t_ef_cd_ab_rho3 = np.tensordot(uef_nu[:,:4], t_cd_ab_nu_rho3, axes=(1, 2))
    t_gh_ef_cd_ab3 = np.tensordot(ugh_rho[:,:4], t_ef_cd_ab_rho3, axes=(1, 3))
    t_ab_cd_ef_gh3 = t_gh_ef_cd_ab3.transpose(3, 2, 1, 0)

    #tree network
    t_ximu_nurho = t_xi_mu_nu_rho.reshape(2**4, 2**4)
    u_ximu_eta, d_eta, u_eta_nurho = np.linalg.svd(t_ximu_nurho, full_matrices=False)

    #move singular value
    u_eta_nu_rho = (np.diag(d_eta) @ u_eta_nurho).reshape(2**4, 2**2, 2**2)
    y_eta_nu_gh= np.tensordot(u_eta_nu_rho, ugh_rho, axes=(2, 1))
    y_etanu_gh = y_eta_nu_gh.reshape(2**6, 2**2)
    u_etanu_rho, d_rho, v_rho_gh = np.linalg.svd(y_etanu_gh, full_matrices=False)

    #gage transformation
    u = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    uab_xi = np.tensordot(uab_xi, u, axes=(1, 0))
    u_xi_mu_eta = u_ximu_eta.reshape(2**2, 2**2, 2**4)
    u_xi_mu_eta = np.tensordot(u.T, u_xi_mu_eta, (1, 0))

    #Check again
    u_eta_nu_rho = u_etanu_rho.reshape(2**4, 2**2, 2**2)
    u_ab_mu_eta = np.tensordot(uab_xi, u_xi_mu_eta, axes=(1, 0))
    u_ab_eta_cd = np.tensordot(u_ab_mu_eta, ucd_mu, axes=(1, 1))
    #u_ab_cd_eta = np.tensordot(u_ab_eta_cd, np.diag(d_eta), axes=(1, 0))
    #u_nu_rho_ab_cd = np.tensordot(u_eta_nu_rho, u_ab_cd_eta, axes=(0, 2))
    #u_rho_ab_cd_ef = np.tensordot(u_nu_rho_ab_cd, uef_nu, axes=(0, 1))
    #t_ab_cd_ef_gh = np.tensordot(u_rho_ab_cd_ef, ugh_rho, axes=(0, 1))

    u_nu_rho_ab_cd = np.tensordot(u_eta_nu_rho, u_ab_eta_cd, axes=(0, 1))
    u_rho_ab_cd_ef = np.tensordot(u_nu_rho_ab_cd, uef_nu, axes=(0, 1))
    t_ab_cd_ef_gh = np.tensordot(u_rho_ab_cd_ef, np.diag(d_rho)@v_rho_gh, axes=(0, 0))

    #t_cd

    pass
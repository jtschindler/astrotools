import numpy as np
import matplotlib.pyplot as plt


def mag_double_power_law(M, phi_star, M_star, alpha, beta):

    A = pow(10, 0.4 * (alpha + 1) * (M - M_star))
    B = pow(10, 0.4 * (beta + 1) * (M - M_star))

    return phi_star / (A + B)


def lum_double_power_law(L, phi_star, L_star, alpha, beta):

    A = pow((L / L_star), alpha)
    B = pow((L / L_star), beta)

    return phi_star / (A + B)


class QLF:

    def __init__(self):

        print('QLF')
        #self.setup = self.setup_qlf_Ro13()


    def eval(self):

        if self.type == 0:
            #magnitude case
            return mag_double_power_law(self.x, self.phi_star, self.x_star, \
                                        self.alpha, self.beta)

        if self.type == 1:
            #luminosity case
            return lum_double_power_law( pow(10.0, self.x ), self.phi_star, self.x_star, \
                                        self.alpha, self.beta)

        if self.type == 2:
            # single power law magnitude case
            return self.phi_star * \
                    pow(10, -0.4 * ( self.x - self.x_star ) * ( self.beta + 1 ))

        if self.type == 3:
            # single power law luminosity case
            return self.phi_star * pow(10,self.beta)

    def setup_qlf_Ro13(self):

        self.name = "Ross2013"
        self.band = "i"
        self.band_wavelength = 445 # in nm
        self.type = 0 # 0 = magnitudes, 1 = luminosities
        self.k_correction = - 0.5964 # k-corrects magnitude to z=0

        self.z_min = 0.3 # lower redshift limit of data for the QLF fit
        self.z_max = 3.5 # upper redshift limit of data for the QLF fit

        self.x_max = -11.8492
        self.x_min = -38.8381

        self.x = -26 # default i-band magnitude value
        self.z = 1.0 # default redshift value

        # best fit values Table 8; first PLE model
        self.log_phi_star_1 = [ -5.96, 0.02, 0.06]
        self.Mi_star_1       = [ -22.85, 0.05, 0.11]
        self.alpha_1        = [ -1.16, 0.02, 0.04]
        self.beta_1         = [-3.37, 0.03, 0.05]
        self.k1           = [ 1.241, 0.01, 0.028]
        self.k2           = [ -0.249, 0.006, 0.017]


        # best fit values Tables 8; second LEDE model
        self.log_phi_star_2 = [-5.83, 0.15, 0.25]
        self.Mi_star_2       = [-26.49, 0.34, 0.24] #k_correction added here to M_i(z=0)
        self.alpha_2        = [-1.31, 0.52, 0.19]
        self.beta_2         = [-3.45, 0.35, 0.21]
        self.c1           = [-0.675, 0.151, 0.011]
        self.c2           = [-0.875, 0.069, 0.3]

    def update_qlf_Ro13(self,Mi,z):

        if( z <= 2.2 ):
            #Eqn. 8
            Mi_star = self.Mi_star_1[0] - 2.5 * ( self.k1[0]*z + self.k2[0]*z**2)
            log_phi_star = self.log_phi_star_1[0]
            alpha = self.alpha_1[0]
            beta = self.beta_1[0]

        else:
            # Eqn. 12
            log_phi_star = self.log_phi_star_2[0] + self.c1[0] * ( z - 2.2 )
            Mi_star = self.Mi_star_2[0] + self.c2[0] * ( z - 2.2 )
            alpha = self.alpha_2[0]
            beta = self.beta_2[0]

        self.z = z
        self.x = Mi

        self.phi_star = pow(10.0,log_phi_star)
        self.x_star = Mi_star
        self.alpha = alpha
        self.beta = beta


    def setup_qlf_Fan99(self):
        self.name = "Fan99"
        self.band = "B"
        self.band_wavelength = 445 # in nm
        self.type = 0 # 0 = magnitudes, 1 = luminosities
        self.k_correction = 0 # k-corrects magnitude to z=0

        self.z_min = 0 # lower redshift limit of data for the QLF fit
        self.z_max = 4 # upper redshift limit of data for the QLF fit

        self.x_max = -12
        self.x_min = -38

        self.x = -26 # default magnitude value
        self.z = 1.0 # default redshift value

        self.M_star_0 = - 27.10
        self.phi_star_0 = 8.22e-7
        self.alpha = 0.5
        self.z_star = 2.75
        self.sigma_z = 0.93

        self.beta_l = -1.64
        self.beta_h = -3.52

        self.M_star_hz = -27.72
        self.beta_hz = -2.87
        self.phi_star_hz = 2.42e-7


        # self.update() = self.update_qlf_Fan99()


    def update_qlf_Fan99(self,M,z):

        if z <= 4.0 :
            M_star = self.M_star_0 + 2.5 * (1-0.5) *np.log10(1.+z) + \
                                1.086 * (z-self.z_star)**2 / (2 * self.sigma_z**2)

            self.x = M
            self.z = z

            self.phi_star = self.phi_star_0
            self.x_star = M_star
            self.alpha = self.beta_l
            self.beta = self.beta_h

        if z > 4.0 :

            self.type = 2
            self.x_star = self.M_star_hz + 0.57 * z
            self.beta = self.beta_hz
            self.phi_star = self.phi_star_hz
            self.x = M
            self.z = z


    def setup_qlf_PD13(self):
        self.name = "PD2013"
        self.band = "i"  # converted to i-band, see the update function
        self.band_wavelength = 445  # in nm
        self.type = 0 # 0 = magnitudes, 1 = luminosities
        self.k_correction = - 0.5964 # k-corrects magnitude to z=0

        self.z_min = 0.68 # lower redshift limit of data for the QLF fit
        self.z_max = 2.6 # upper redshift limit of data for the QLF fit

        self.x_max = -11.7812
        self.x_min = -38.7701

        self.x = -25 # default magnitude value
        self.z = 1.0 # default redshift value

        # best fit values Table 7
        self.log_phi_star = [ -5.89, 0.03, 0.03]
        self.Mg_star_z =    [ -26.36, 0.06, 0.06]
        self.alpha_l =      [ -3.5, 0.05, 0.05]
        self.beta_l =       [ -1.43, 0.03, 0.03]
        self.k1l =          [ 0.03, 0.02, 0.02]
        self.k2l =          [ -0.34, 0.01, 0.01]
        self.alpha_h =      [ -3.19, 0.07, 0.07]
        self.beta_h =       [ -1.17, 0.05, 0.05]
        self.k1h =          [ -0.35, 0.13, 0.13]
        self.k2h =          [ -0.02, 0.14, 0.14]
        self.z_p = 2.2


    def update_qlf_PD13(self,Mi,z):

        Mg = Mi + 0.51* (-0.5)

        if z < self.z_p:
            alpha = self.alpha_l[0]
            beta = self.beta_l[0]
            k1 = self.k1l[0]
            k2 = self.k2l[0]
        else :
            alpha = self.alpha_h[0]
            beta = self.beta_h[0]
            k1 = self.k1h[0]
            k2 = self.k2h[0]

        self.x = Mg
        self.z = z

        self.phi_star = pow(10,self.log_phi_star[0])
        # Eq. 11 of Palanque - Delabrouille 2013
        self.x_star = self.Mg_star_z[0] \
                            - 2.5 * ( k1 * (z-self.z_p) + k2 * (z-self.z_p)**2 )
        self.alpha = alpha
        self.beta = beta

    def setup_qlf_Ri06(self):

        self.name = "Richards2006"
        self.band = "i"
        self.band_wavelength = 445 # in nm
        self.type = 3 # 0 = magnitudes, 1 = luminosities # 2 = single power law
        self.k_correction = - 0.5964 # k-corrects magnitude to z=0

        self.z_min = 0.68 # lower redshift limit of data for the QLF fit
        self.z_max = 2.6 # upper redshift limit of data for the QLF fit

        self.x_max = -11.7812
        self.x_min = -38.7701

        self.x = -25 # default magnitude value
        self.z = 1.0 # default redshift value




        # best fit values from the second row of Table 7
        self.A1 = [ 0.83, 0.01, 0.01]
        self.A2 = [ -0.11, 0.01, 0.01]
        self.B1 = [ 1.43, 0.04, 0.04]
        self.B2 = [ 36.64, 0.1, 0.1]
        self.B3 = [ 34.39, 0.26, 0.26]
        self.M_star = -26
        self.z_ref = 2.45
        self.log_phi_star = -5.70

    def update_qlf_Ri06(self,Mi,z):

        psi = np.log10( ( 1. + z ) / ( 1. + self.z_ref ) )

        mu = Mi - ( self.M_star + self.B1[0] * psi + \
                                self.B2[0] * psi**2 + self.B3[0] * psi**3 )

        self.phi_star = pow(10,self.log_phi_star)
        self.beta = mu * ( self.A1[0] + self.A2[0] * ( z - 2.45 ) )

        self.x = Mi
        self.z = z



    def setup_qlf_Ho07(self):
        self.name = "Hopkins2007"
        self.band = "bolometric"
        self.band_wavelength = None # in nm
        self.type = 1 # 0 = magnitudes, 1 = luminosity double power law

        self.z_min = 0 # lower redshift limit of data for the QLF fit
        self.z_max = 4.5 # upper redshift limit of data for the QLF fit

        self.x_max = 18 #log(L_bol)
        self.x_min = 8 #log(L_bol)

        self.x = 12 # default magnitude value
        self.z = 1.0 # default redshift value

        # best fit values Table 7
        self.log_phi_star = [ -4.825, 0.06, 0.06]
        self.logL_star =    [ 13.036, 0.043, 0.043]
        self.gamma_1 =      [ 0.417, 0.055, 0.055]
        self.gamma_2 =      [ 2.174, 0.055, 0.055]
        self.kl1 =          [ 0.632, 0.077, 0.077]
        self.kl2 =          [ -11.76, 0.38, 0.38]
        self.kl3 =          [ -14.25, 0.8, 0.8]
        self.kg1 =          [ -0.623, 0.132, 0.132]
        self.kg2_1 =        [ 1.46, 0.096, 0.096]
        self.kg2_2 =        [ -0.793, 0.057, 0.057]

        self.z_ref = 2.0


    def update_qlf_Ho07(self,logL,z):

        # Equation 10
        xi = np.log10( (1+z) / (1+self.z_ref) )

        self.z = z
        self.x = logL

        # Equation 9
        self.x_star = self.logL_star[0] + self.kl1[0] * xi + self.kl2[0] * xi**2 \
                    + self.kl3[0] * xi**3
        self.x_star = pow(10,self.x_star)

        # Equation 17
        self.alpha = self.gamma_1[0] * pow(10,self.kg1[0]*xi)

        # Equation 19
        self.beta = self.gamma_2[0] * 2 / \
                    ( pow(10,self.kg2_1[0]*xi) + (pow(10,self.kg2_2[0]*xi)))


        # note pg. 744 right column bottom
        if ((self.beta < 1.3) & (z > 5.0)):
            self.beta = 1.3

        self.phi_star = pow(10,self.log_phi_star[0])


    def setup_qlf_PD16(self):
        self.name = "PD2016"
        self.band = "i"  # converted to i-band, see the update function
        self.band_wavelength = 445  # in nm
        self.type = 0 # 0 = magnitudes, 1 = luminosities
        self.k_correction = 0 # k-corrects magnitude to z=0
        # self.k_correction = - 0.5964 # k-corrects magnitude to z=0


        self.z_min = 0.68 # lower redshift limit of data for the QLF fit
        self.z_max = 4.0 # upper redshift limit of data for the QLF fit

        self.x_max = -11.7812
        self.x_min = -38.7701

        self.x = -25 # default magnitude value
        self.z = 1.0 # default redshift value

        # best fit values Table 7
        self.log_phi_star = [-5.93, 0.09, 0.09]
        self.Mg_star_z =      [-22.25, 0.49, 0.49]
        self.alpha_0 =        [-3.89, 0.23, 0.23]
        self.beta_0 =         [-1.47, 0.06, 0.06]
        self.k1 =           [1.59, 0.28, 0.28]
        self.k2 =           [-0.36, 0.09, 0.09]
        self.c1a =          [-0.46, 0.10, 0.10]
        self.c1b =          [-0.06, 0.10, 0.10]
        self.c2 =          [-0.14, 0.17, 0.17]
        self.c3 =          [0.32, 0.23, 0.23]
        self.zp = 0

    def update_qlf_PD16(self, Mi, z):

        Mg = Mi + 0.51 * (-0.5)

        x_star_zp = self.Mg_star_z[0]
        log_phi_star_zp = self.log_phi_star[0]
        # x_star_zp = self.Mg_star_z[0] - 2.5 * (self.k1[0] * (0 - self.zp) +
        # self.k2[0] * ( 0 - self.zp)** 2)
        # log_phi_star_zp = self.log_phi_star[0] - self.c1a[0] * (0 - self.zp) \
        #                   - self.c1b[0] * (0 - self.zp) ** 2

        if z < 2.2:
            # use the PLE model
            self.phi_star = pow(10, log_phi_star_zp)
            self.alpha = self.alpha_0[0]
            self.beta = self.beta_0[0]

            # Eq. 7 of Palanque - Delabrouille 2016
            self.x_star = x_star_zp \
                          - 2.5 * (self.k1[0] * (z - self.zp) + self.k2[0] * (
                          z - self.zp)** 2)

            # print(self.phi_star, self.x_star, self.alpha, self.beta, z)

        else:
            # use the LEDE model
            # Eq. 8,9 of Palanque - Delabrouille 2016
            x_star_zp = self.Mg_star_z[0] \
                          - 2.5 * (self.k1[0] * (2.2) + self.k2[0] * (
                          2.2)** 2)
            log_phi_star_zp = self.log_phi_star[0]

            self.phi_star = log_phi_star_zp
            self.phi_star += self.c1a[0] * (z - 2.2) + self.c1b[0] * (
                        z - 2.2)** 2
            self.phi_star = pow(10, self.phi_star)
            self.x_star = x_star_zp + self.c2[0] * (z - 2.2)
            self.alpha = self.alpha_0[0] + self.c3[0] * (z - 2.2)
            self.beta = self.beta_0[0]

            # print(self.phi_star, self.x_star, self.alpha, self.beta, z)

        self.x = Mg
        self.z = z


    def setup_qlf_K19(self):
        self.name = "K2019"
        self.band = "M1450"  # converted to i-band, see the update function
        self.band_wavelength = 145  # in nm
        self.type = 0  # 0 = magnitudes, 1 = luminosities
        self.k_correction = 0  # k-corrects magnitude to z=0
        # self.k_correction = - 0.5964 # k-corrects magnitude to z=0

        self.z_min = 0.3  # lower redshift limit of data for the QLF fit
        self.z_max = 10.0  # upper redshift limit of data for the QLF fit

        self.x_max = -11.7812
        self.x_min = -38.7701

        self.x = -25  # default magnitude value
        self.z = 1.0  # default redshift value

        # best fit values
        self.c0_0 = -6.942
        self.c0_1 = 0.629
        self.c0_2 = -0.086
        self.c1_0 = -15.038
        self.c1_1 = -7.046
        self.c1_2 = 0.772
        self.c1_3 = -0.030
        self.c2_0 = -2.888
        self.c2_1 = -0.383
        self.c3_0 = -1.602
        self.c3_1 = -0.082




    def update_qlf_K19(self, M1450, z):

        x = 1 + z

        logPhiStar = np.polynomial.chebyshev.chebval(x, [self.c0_0,
                                                         self.c0_1, self.c0_2])
        self.phi_star = np.pow(10, logPhiStar)

        self.x_star = np.polynomial.chebyshev.chebval(x, [self.c1_0,
                                                           self.c1_1, self.c1_2, self.c1_3])

        self.alpha = np.polynomial.chebyshev.chebval(x, [self.c2_0, self.c2_1])

        self.beta = self.c3_0 + self.c3_1 * x

# Ro13 = QLF()
# Ro13.setup_qlf_Ro13()
# Ri06 = QLF()
# Ri06.setup_qlf_Ri06()
# # # Fan99 = QLF()
# # # Fan99.setup_qlf_Fan99()
# # #
# # #
# M_i = np.arange(-30,-17.5,0.5)
# phi = np.zeros(len(M_i))
# phi2 = np.zeros(len(M_i))
# for i in range(len(M_i)):
#     Ro13.update_qlf_Ro13(M_i[i],3.2)
#     Ri06.update_qlf_Ri06(M_i[i],3.2)
#     phi[i] = Ro13.eval()
#     phi2[i] = Ri06.eval()
# # #
# # #
# plt.plot(M_i, np.log10(phi),'r')
# plt.plot(M_i, np.log10(phi2),'g')
# # #
# plt.axis([-18,-32,-10,-4])
# plt.show()

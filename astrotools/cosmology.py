import numpy as np
from scipy import integrate

from .constants import *

#adapted from the Cosmology class from Brant Robertson

class Cosmology:

    def __init__(self):

        # Hubble constant at z=0 in h km/s/Mpc
        self.H_0 = 100.0

        #  Inverse of Hubble constant in h^-1 Gyr
        self.H_inv = 1 / (self.H_0 * 1.0e3 / mpc * year_in_sec * 1.0e9)

        # critical density in h^2 Msun / Mpc^3
        self.rho_crit = 3.0*self.H_0*self.H_0/(8*pi*G_cosmo*1.0e-3)

        # critical density in h^2 g / cm^3
        self.rho_c = self.rho_crit * Msun_cgs / pow(mpc_cgs,3)


    def E(self,z):
    #  returns unit free Hubble parameter at redshift z

        return np.sqrt( self.Omega_m*pow(1.+z,3) + \
                        self.Omega_k*pow(1.+z,2) + \
                        self.Omega_r*pow(1.+z,4) + \
                        self.Omega_l )

    def H(self,z):
        # returns Hubble parameter at redshift z
        # in h km/s /Mpc

        return self.H_0*self.E(z)

    def rho_m(self,z):
        # Physical matter desnity at redshift z
        # in h^2 Msun/Mpc^3

        return self.Omega_m*self.rho_crit*pow(1.+z,3)


    def distance_modulus(self,z):
        # Return the distance modulus to redshift z in delta mag

        ld_in_pc = 1.0e6*self.luminosity_distance(z)/self.h


        return 5.0*(np.log10(ld_in_pc)-1.0)

    def k_correction(self,z,a_nu):
        # Hogg 1999, eq. 27
        return -2.5 * ( 1 + a_nu ) *  np.log10(1.+z)

    #---------------------------------------------------------------------------
    # Distances and Volume
    #---------------------------------------------------------------------------

    def hubble_distance(self):
        # in h^-3 Mpc^3

        return  c / 1000 / self.H_0

    def hubble_volume(self):
        # in h^-3 Mpc^3

        return self.hubble_distance()**3

    def comoving_distance(self, z_min, z_max):
        # comoving distance in h^-1 Mpc

        return ((c/1.0e3)/self.H_0)* \
                integrate.quad(self.comoving_distance_integrand,z_min,z_max)[0]

    def comoving_distance_integrand(self,z):
        # integrand for the comoving distance Eqn. 15 in Hogg 1999

        return 1./np.sqrt( self.Omega_m*pow(1.+z,3) + \
                            self.Omega_k*pow(1.+z,2) + \
                            self.Omega_r*pow(1.+z,4) + \
                            self.Omega_l )

    def comoving_volume(self, z_min, z_max):
        #returns the comoving volume in h^-3 Mpc^3

        z_max_volume = pow(self.comoving_distance(0,z_max),3)
        z_min_volume = pow(self.comoving_distance(0,z_min),3)

        return 4.0*pi/3.0 * (z_max_volume - z_min_volume)

    def comoving_volume_integrand(self, z):

        # Dimensionless Hubble parameter
        E_z = np.sqrt( self.Omega_m*pow(1.+z,3) + \
                            self.Omega_k*pow(1.+z,2) + \
                            self.Omega_r*pow(1.+z,4) + \
                            self.Omega_l )

        # Hubble Volume
        D_H = self.hubble_distance()

        # Angular diameter distance
        D_A = self.angular_distance(z)


        return 4. * pi * D_H * ( 1. + z )**2 * D_A**2 / E_z


    def integrated_comoving_volume(self, z_min, z_max):

        return integrate.quad(self.comoving_volume_integrand, z_min, z_max)[0]


    def luminosity_distance(self, z):
        #luminosity distance to redshift z in h^-1 Mpc

        return self.comoving_distance(0,z)*(1.+z)


    def angular_distance(self,z):
        #angular distance to redshift z in h^-1 Mpc

        return self.comoving_distance(0,z)/(1.+z)


    #---------------------------------------------------------------------------
    # 3D distances between two sources
    #---------------------------------------------------------------------------

    def distance_3d(self, z1, z2, theta):
        """ Calculate the 3D distance between two sources with redshifts z1, z2
        and angle theta"""

        # see Liske 2000, eqs. 3-7
        # this code only holds for a flat cosmology

        # comoving distances
        r1 = self.comoving_distance(0,z1)
        r2 = self.comoving_distance(0,z2)

        A = ( (r2+r1) / 2. )**2 * np.sin(np.deg2rad(theta / 2.))**2

        B = ( (r2-r1) / 2. )**2 * np.cos(np.deg2rad(theta / 2.))**2

        r2_prime = np.sqrt(A + B) * 2

        # proper distance in h^-1 Mpc
        return ( 1. / (1. + z1) ) * r2_prime



    #---------------------------------------------------------------------------
    # Times and Ages
    #---------------------------------------------------------------------------

    def universal_age(self, z):
        # age of the universe up to redshift of z in h^-1

        return self.H_inv * \
                integrate.quad(self.universal_age_integrand,z,np.inf)[0]

    def universal_age_integrand(self,z):
        #integrand to calculate the age of the universe

        return 1./(1.+z) \
                /np.sqrt( self.Omega_m*pow(1.+z,3) + \
                        self.Omega_k*pow(1.+z,2) + \
                        self.Omega_r*pow(1.+z,4) + \
                        self.Omega_l )


    #---------------------------------------------------------------------------
    # Setting the Cosmology
    #---------------------------------------------------------------------------


    def set_cosmology(self,h, Omega_l, Omega_m, Omega_b, Omega_k, Omega_r, n_s, \
                        sigma_8, T_cmb):

        # Hubble parameter
        self.h = h

        # Omega values
        self.Omega_l = Omega_l
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.Omega_c = Omega_m - Omega_b
        self.Omega_k = Omega_k
        self.Omega_r = Omega_r

        # Spectral slope
        self.n_s = n_s

        # Power spectrum normalization
        self.T_cmb = T_cmb


    def set_zentner2007_cosmology(self):

        self.cosmology_model_name = "Zentner2007"

        self.set_cosmology(h = 0.7, \
                      Omega_l = 0.7, \
                      Omega_m = 0.3, \
                      Omega_b = 0.022/pow(0.7,2), \
                      Omega_k = 0.0, \
                      Omega_r = 4.15e-5/(0.7*0.7), \
                      n_s = 1.0, \
                      sigma_8 = 0.93, \
                      T_cmb = 2.725)


    def set_fan1999_cosmology(self):

        self.cosmology_model_name = "Fan1999"

        self.set_cosmology(h = 0.5, \
                      Omega_l = 0.7, \
                      Omega_m = 0.3, \
                      Omega_b = 0.022/pow(0.7,2), \
                      Omega_k = 0.0, \
                      Omega_r = 4.15e-5/(0.7*0.7), \
                      n_s = 1.0, \
                      sigma_8 = 0.93, \
                      T_cmb = 2.725)


# TEST COSMOLOGY
# Cosmo = Cosmology()
#
# Cosmo.set_zentner2007_cosmology()
#
# print (Cosmo.H(2))
# print (Cosmo.E(2))
# print (Cosmo.rho_m(2))
# print (Cosmo.comoving_distance(0.0,2.0)/0.7)
# print (Cosmo.luminosity_distance(2.0)/0.7)
# print (Cosmo.angular_distance(2.0)/0.7)
# print (Cosmo.hubble_volume())
# print (Cosmo.comoving_volume(0,2.0)/0.7**3)
# print (Cosmo.integrated_comoving_colume(0,2.0)/0.7**3)
# print (Cosmo.universal_age(2.0)/0.7)

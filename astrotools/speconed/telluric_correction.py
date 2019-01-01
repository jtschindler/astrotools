
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

from lmfit import Model

import speconed as sod

import scipy.constants as const

def planck_function_wav(wav,T):
    """ Blackbody as a function of wavelength (Angstroem) and temperature (K).

    returns units of erg/s/cm^2/Angstroem/Steradian
    """

    wav_m = wav * 1e-10

    return  2*const.h*const.c**2 / (wav_m**5 * (np.exp(const.h*const.c / (wav_m*const.k*T)) - 1)) * 1e+7 / 1e+20




science_spectrum = sod.SpecOneD()
science_spectrum.read_from_fits('Q1652_comb_trimmed.fits')
telluric_spectrum = sod.SpecOneD()
telluric_spectrum.read_from_fits('HD287515_comb_trimmed.fits')

# uka0 = np.genfromtxt('uka0v.dat')

uka0 = np.genfromtxt('/home/jt/Downloads/vegallpr25.2000')

wav = np.arange (10000,25000)
planck = planck_function_wav(wav,9700)



telluric_model = sod.SpecOneD(dispersion=uka0[:,0], flux=uka0[:,1])

tell_min = telluric_spectrum.dispersion.min()
tell_max = telluric_spectrum.dispersion.max()

telluric_model.trim_disperion(lo_limit=tell_min, up_limit=tell_max, mode='wav')

telluric_model.flux = telluric_spectrum.flux.mean () / telluric_model.flux.mean() * telluric_model.flux

# plt.plot(wav,planck+60000)
plt.plot(telluric_spectrum.dispersion, telluric_spectrum.flux)
plt.plot(telluric_model.dispersion, telluric_model.flux+20000)
plt.plot(science_spectrum.dispersion, science_spectrum.flux)

plt.show()

# References Vacca et al. 2003, Maiolino et al. 1996
# model spectrum needs to be shifted, scaled and reddened; altering depth of H lines, and resampling to wavelength scale
# radial velocity if known could be entered

# 1) determine continuum to normalize observed A0V spectrum around a absorption line
# 2) cross correlate normalized observed A0V with normalized A0V close to spectral absorption line
# 3) Unnormalized (full) model is then shifted by estimated radial velocity and scaled to match observed magnitudes of the A0V star



# IRAF Procedure
# 0.1) Each spectrum is prepared as follows, a large sacle continuum is fit
# using iterative sigmaclipping and subtracted from the spectrum
# efficiently flattening the spectrum.
# 1) cross correlation of input and calibration spectrum to determine shift in dispersion/x

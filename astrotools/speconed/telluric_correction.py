
import numpy as np
from numpy import sqrt, pi, exp
import matplotlib.pyplot as plt

from lmfit import Model
import sys
from astrotools.speconed import speconed as sod

from astrotools.speconed import interactive as inter

import scipy.constants as const

def planck_function_wav(wav,T):
    """ Blackbody as a function of wavelength (Angstroem) and temperature (K).

    returns units of erg/s/cm^2/Angstroem/Steradian
    """

    wav_m = wav * 1e-10

    return  2*const.h*const.c**2 / (wav_m**5 * (np.exp(const.h*const.c / (wav_m*const.k*T)) - 1)) * 1e+7 / 1e+20


inlist = {'science':'J034151_L1_combined.fits',
          'tellstar':'HD24000_L1_combined.fits',
          'extinction_science':0.4528,
          'extinction_telluric':0.7460,
          'R_V':3.1,
          'extinction_law':'fm07',
          'tellstarmodel':'uka0v.fits',
          'tellcorr_name':'tellcorr_L1_2.fits'}

inlist2 = {'science':'J034151_L2_combined.fits',
          'tellstar':'HD24000_L2_combined.fits',
          'extinction_science':0.4528,
          'extinction_telluric':0.7460,
          'R_V':3.1,
          'extinction_law':'fm07',
          'tellstarmodel':'uka0v.fits',
          'tellcorr_name':'tellcorr_L2.fits'}



def telluric_correction(inlist, interactive=True, fileformat='fits'):


    # Read in spectra
    science = sod.SpecOneD()
    tellstar = sod.SpecOneD()
    if fileformat=='fits':
        science.read_from_fits(inlist['science'])
        tellstar.read_from_fits(inlist['tellstar'])
    elif fileformat=='hdf':
        science.read_from_hdf(inlist['science'])
        tellstar.read_from_hdf(inlist['tellstar'])

    print(tellstar.flux_err, science.flux_err)

    tellstarmodel = sod.SpecOneD()
    tellstarmodel.read_from_fits(inlist['tellstarmodel'])
    tellstarmodel.flux_err = np.zeros(len(tellstarmodel.dispersion))



    # Deredden science and telluric spectrum

    science.deredden(inlist['extinction_science'],
                     inlist['R_V'],
                     extinction_law=inlist['extinction_law'],
                     inplace=True)

    tellstar.deredden(inlist['extinction_telluric'],
                     inlist['R_V'],
                     extinction_law=inlist['extinction_law'],
                     inplace=True)




    # 1) Build telluric

    if interactive:
        app = inter.QtWidgets.QApplication(sys.argv)
        form = inter.SpecOneDGui(spec_list=[tellstar, tellstarmodel],
                                 mode="divide", fileformat='hdf',
                                 copy_flux_err='first')
        form.show()
        app.exec_()

    # Read in telluric correction
    tellcorr = sod.SpecOneD()
    if fileformat=='fits':
        tellcorr.read_from_fits(inlist['tellcorr_name'])
    elif fileformat=='hdf':
        tellcorr.read_from_hdf(inlist['tellcorr_name'])
    if interactive:
        app = inter.QtWidgets.QApplication(sys.argv)
        form = inter.SpecOneDGui(spec_list=[science, tellcorr], mode="divide",
                                 fileformat='hdf', copy_flux_err='No')
        form.show()
        app.exec_()

# telluric_correction(inlist)

# need to test this

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

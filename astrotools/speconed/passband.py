
import sys
import os
import numpy as np
from .speconed import SpecOneD
from .speconed import datadir


import matplotlib.pyplot as plt


from astropy import constants as const


class PassBand(SpecOneD):

    def __init__(self, dispersion=None, flux=None, flux_err=None, header=None,
                 passband_name=None, unit=None):

        if passband_name is not None:
            self.load_passband(passband_name)
        else:
            super(PassBand, self).__init__(self, dispersion=dispersion,
                                           flux=flux, flux_err=flux_err,
                                           header=header, unit=unit)

    def load_passband(self, passband_name):


        passband_data = np.genfromtxt(datadir+'passbands/'+passband_name+'.dat')



        wavelength = passband_data[:, 0]
        throughput = passband_data[:, 1]

        # Change wavelength to Angstroem for all passbands
        filter_group = passband_name.split('-')[0]

        if filter_group == "WISE":
            # micron to Angstroem
            wavelength = wavelength * 10000.

        elif filter_group == "LSST":
            # nm to Angstroem
            wavelength = wavelength * 10.


        self.dispersion = wavelength
        self.flux = throughput

        self.raw_dispersion = wavelength
        self.raw_flux = throughput

        self.flux_err = None
        self.raw_flux_err = None

        self.header = None

        self.mask = np.ones(self.dispersion.shape, dtype=bool)

        self.unit = 'f_lam'
        self.model_spectrum = None
        self.fit_output = None


    def to_wavelength(self):

        if self.unit != 'f_nu':
            raise ValueError('Dispersion must be in frequency (Hz)')

        self.dispersion = (const.c.value * 1e+10) / self.dispersion

        self.flux = np.flip(self.flux, axis=0)
        self.dispersion = np.flip(self.dispersion, axis=0)

        self.unit = 'f_lam'

    def to_frequency(self):

        if self.unit != 'f_lam':
            raise ValueError('Dispersion must be in wavelength (Angstroem)')

        self.dispersion = (const.c.value * 1e+10) / self.dispersion

        self.flux = np.flip(self.flux, axis=0)
        self.dispersion = np.flip(self.dispersion, axis=0)

        self.unit = 'f_nu'

    def plot(self, show_flux_err=False, show_raw_flux=False, mask_values=True):

        """Plot the spectrum

        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

        if show_flux_err:
            self.ax.plot(self.dispersion[mask], self.flux_err[mask], 'grey', lw=1)
        if show_raw_flux:
            self.ax.plot(self.raw_dispersion[mask], self.raw_flux[mask], 'grey', lw=3)

        self.ax.plot(self.dispersion[mask], self.flux[mask], 'k', linewidth=1)

        if self.unit=='f_lam':
            self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Filter\ Transmission}$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Filter\ Transmission}$', fontsize=15)

        else :
            raise ValueError("Unrecognized units")

        # If a model spectrum exists, print it
        if self.model_spectrum:
            model_flux = self.model_spectrum.eval(self.model_pars, x=self.dispersion)
            self.ax.plot(self.dispersion[mask], model_flux[mask])

        if self.fit_output:
            self.ax.plot(self.dispersion[mask], self.fit_output.best_fit[mask])

        plt.show()

import numpy as np
from .speconed import SpecOneD

from matplotlib import rc
import matplotlib.pyplot as plt

class FlatSpectrum(SpecOneD):

    def __init__(self, flat_dispersion, unit='f_nu'):

        try:
            flat_dispersion = np.array(flat_dispersion)
            if flat_dispersion.ndim != 1:
                raise ValueError("Flux dimension is not 1")
        except ValueError:
            print("Flux could not be converted to 1D ndarray")

        if unit == 'f_lam':
            fill_value = 3.631e-9
        if unit == 'f_nu':
            fill_value = 3.631e-20

        self.flux = np.full(flat_dispersion.shape, fill_value)
        self.raw_flux = self.flux
        self.dispersion = flat_dispersion
        self.raw_dispersion = self.dispersion
        self.unit = unit

    def plot(self):

        """Plot the spectrum

        """

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

        self.ax.plot(self.dispersion, self.flux, 'k', linewidth=1)

        if self.unit=='f_lam':
            self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

        else :
            raise ValueError("Unrecognized units")


        plt.show()

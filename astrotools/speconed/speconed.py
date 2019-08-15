#!/usr/bin/env python

"""The SpecOneD module.

This module introduces the SpecOneD class, it's functions and the related
FlatSpectrum and QuasarSpectrum classes.
The main purpose of the SpecOneD class and it's children classes is to provide
python functionality for the manipulation of 1D spectral data in astronomy.


Example
-------
Examples are provided in a range of jupyter notebooks accompanying the module.

Notes
-----
    This module is in active development and its contents will therefore change.


Attributes
----------
datadir : str
    The path to the data directory formatted as  a string.
"""

import sys
import os
import numpy as np
import scipy as sp
import extinction as ext
import pandas as pd

from astropy.io import fits
from astropy.constants import c
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel

from matplotlib import rc
import matplotlib.pyplot as plt

from numpy.polynomial import Legendre, Chebyshev, Polynomial

from lmfit import Model
from lmfit.models import ExponentialModel, GaussianModel, LinearModel


datadir = os.path.split(__file__)[0]
datadir = os.path.split(datadir)[0] + '/data/'


def gaussian(x, amp, cen, sigma, shift):
    """ 1-D Gaussian """
    central = cen + shift

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


class SpecOneD(object):

    """The SpecOneD class stores 1D spectral information and allows it's
    manipulation with built-in functions.

    Attributes
    ----------
    raw_dispersion : ndarray
        A 1D array containing the original spectral dispersion data in 'float'
        type.
    raw_flux : ndarray
        A 1D array containing the original spectral flux data in 'float' type.
    raw_flux_err : ndarray
        A 1D array containing the original spectral flux error data in 'float'
        type.
    dispersion : ndarray
        The 1D array containing the sp`ectral dispersion data in 'float' type.
    flux : ndarray
        A 1D array containing the spectral flux data in 'float' type.
    flux_err : ndarray
        A 1D array containing the spectral flux error data in 'float' type.
    mask : ndarray
        A 1D mask array (boolean type).
    unit : str
        A string specifying if the if the flux unit is per wavelength 'f_lam'
        or frequency 'f_nu'.
    model_spectrum : obj:
        A lmfit Model object, which allows to fit the spectrum using lmfit
    model_pars : obj:
        A lmfit parameter object containing the parameters for the
        model_spectrum.
    fit_output : obj:
        The resulting fit based on the model_spectrum and the model_pars
    fit_dispersion :
        A 1D array containing the dispersion of the fitted spectrum.
    fit_flux : ndarray
        A 1D arrat containing the flux values for the fitted spectrum.
    header : obj
        The spectral header object, containing additional data with regard to
        the spectrum.

    """

    def __init__(self, dispersion=None, flux=None, flux_err=None, header=None, unit=None, mask=None, raw_dispersion=None, raw_flux=None, raw_flux_err=None):
        """The __init__ method for the SpecOneD class

        Parameters
        ----------
        dispersion : ndarray
            A 1D array providing the dispersion data of the spectrum in
            wavelength or frequency.
        flux : ndarray
            A 1D array providing the flux data for the spectrum.
            lines are supported.
        flux_err : ndarray
            A 1D array providing the error on the spectrum flux
        header : obj:`dict`
            The header object file for the spectrum. This should be a python
            dictionary or a fits format header file.
        unit : str
            A string defining the unit of the flux measurement. Currently
            flux per wavelength or per frequency are supported as the following
            options: 'f_lam' for wavelength, 'f_nu' for frequency.
        mask : ndarray
            A 1D boolean array can be specified to provide a mask for the
            spectrum.

        Raises
        ------
        ValueError
            Raises an error when either the dispersion or flux dimension is not
            or could not be converted to a 1D ndarray.
        ValueError
            Raises an error when the supplied header is not a dictionary.

        """

        # disperion units need to be in Angstroem

        try:
            if flux is None:
                self.raw_flux = flux
            else:
                self.raw_flux = np.array(flux)
                if flux.ndim != 1:
                    raise ValueError("Flux array is not 1D")
        except ValueError:
            print("Flux could not be converted to 1D ndarray")

        try:
            if dispersion is None:
                self.raw_dispersion = dispersion
            else:
                self.raw_dispersion = np.array(dispersion)
                if dispersion.ndim != 1:
                    raise ValueError("Flux dimension is not 1")
        except ValueError:
            print("Flux could not be converted to 1D ndarray")

        self.raw_flux_err = flux_err

        self.flux = self.raw_flux

        self.flux_err = self.raw_flux_err
        self.dispersion = self.raw_dispersion

        if mask is not None:
            self.mask = mask
        elif self.dispersion is None:
            self.mask = None
        else:
            self.mask = np.ones(self.dispersion.shape, dtype=bool)

        self.unit = unit
        self.model_spectrum = None
        self.model_pars = None
        self.fit_dispersion = None
        self.fit_flux = None
        self.fit_output = None


        if header == None:
            self.header = dict()
        else:
            self.header = header


    def read_pypeit_fits(self, filename, unit='f_lam', exten=1):
        """ Read a 1D fits pypeit file to populate the SpecOneD class.

        :param filename: str
            Path/Filename of the pypeit 1D fits file
        :param unit:
            The unit of the flux measurement in the fits file. This defaults
            to flux per wavelength (erg/s/cm^2/Angstroem)
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        self.header = hdu[0].header
        self.unit = unit
        self.dispersion = hdu[exten].data['OPT_WAVE']
        self.raw_dispersion = hdu[exten].data['OPT_WAVE']
        self.flux = hdu[exten].data['OPT_FLAM']*1e-17
        self.raw_flux = hdu[exten].data['OPT_FLAM']*1e-17
        self.flux_err = 1./np.sqrt(hdu[exten].data['OPT_FLAM_IVAR'])*1e-17
        self.raw_flux_err = 1./np.sqrt(hdu[exten].data['OPT_FLAM_IVAR'])*1e-17
        self.flux_sig = hdu[exten].data['OPT_FLAM_SIG']
        self.mask = hdu[exten].data['OPT_MASK']

        if 'TELLURIC' in hdu[exten].columns.names:
            self.telluric = hdu[exten].data['TELLURIC']
        if 'OBJ_MODEL' in hdu[exten].columns.names:
            self.obj_model = hdu[exten].data['OBJ_MODEL']*1e-17


    def read_from_fits(self, filename, unit='f_lam'):
        """Read a 1D fits file to populate the SpecOneD class.

        Parameters
        ----------
        filename : str
            A string providing the path and filename for the fits file.
        unit : str
            The unit of the flux measurement in the fits file. This defaults
            to flux per wavelength (erg/s/cm^2/Angstroem)

        Raises
        ------
        ValueError
            Raises an error when the filename could not be read in.
        """

        # Open the fits file
        try:
            hdu = fits.open(filename)
        except:
            raise ValueError("Filename not found", str(filename))

        # Read in header information
        crval = hdu[0].header['CRVAL1']
        try:
            cd = hdu[0].header['CD1_1']
        except:
            print("CD1_1 keyword not found, using CDELT1")
            cd = hdu[0].header['CDELT1']

        crpix = hdu[0].header['CRPIX1']
        naxis = hdu[0].header['NAXIS1']


        self.unit = unit

        # Read in object flux
        if np.ndim(hdu[0].data) == 3:
            try:
                self.flux = np.array(hdu[0].data[0, 0, :])
                self.flux_err = np.array(hdu[0].data[3, 0, :])
            except:
                self.flux = np.array(hdu[0].data[0,0,:])
                self.flux_err = np.array(hdu[0].data[1,0,:])
        else:
            self.flux = np.array(hdu[0].data[:])
            self.flux_err = None


        # Calculate dispersion axis from header information
        crval = crval - crpix
        self.dispersion = crval + np.arange(naxis) * cd

        self.raw_flux = self.flux
        self.mask = np.ones(self.dispersion.shape, dtype=bool)
        self.raw_flux_err = self.flux_err
        self.raw_dispersion = self.dispersion

        self.header = hdu[0].header

    def save_to_fits(self, filename, comment=None, overwrite = False):
        """Save a SpecOneD spectrum to a fits file.

        Note: This save function does not store flux_errors, masks, fits etc. Only the original header, the dispersion (via the header), and the flux
        are stored.

        Parameters
        ----------
        filename : str
            A string providing the path and filename for the fits file.

        Raises
        ------
        ValueError
            Raises an error when the filename exists and overwrite = False
        """


        hdu  = fits.PrimaryHDU(self.flux)
        hdu.header = self.header

        # Update header information
        crval = self.dispersion[0]
        cd = self.dispersion[1]-self.dispersion[0]
        crpix = 1

        hdu.header['CRVAL1'] = crval
        hdu.header['CD1_1'] = cd
        hdu.header['CDELT1'] = cd
        hdu.header['CRPIX1'] = crpix

        hdu.header['HISTORY'] = '1D spectrum generated with SpecOneD'

        if comment:
            hdu.header['HISTORY'] = comment

        hdul = fits.HDUList([hdu])

        try:
            hdul.writeto(filename, overwrite = overwrite)
        except:
            raise ValueError("Spectrum could not be saved. Maybe a file with the same name already exists and overwrite is False")

    def save_to_hdf(self, filename, comment=None, overwrite=False):

        df = pd.DataFrame(np.array([self.dispersion, self.flux]).T, columns=['dispersion', 'flux'])

        if self.flux_err is not None:
            df['flux_err'] = self.flux_err

        df.to_hdf(filename,'data')


    def save_to_csv(self, filename, format='linetools'):

        data = [self.dispersion, self.dispersion]

        if self.flux_err is not None:
            data.append(self.flux_err)


        if hasattr(self,'telluric'):
            data.append(self.telluric)
        if hasattr(self,'obj_model'):
            data.append(self.obj_model)

        if format=='linetools':

            column_names = ['wave', 'flux']
            if self.flux_err is not None:
                column_names.append('error')

        else:

            column_names = ['wavelength', 'flux']
            if self.flux_err is not None:
                column_names.append('flux_error')

        if hasattr(self, 'telluric'):
            column_names.append('telluric')
        if hasattr(self, 'obj_model'):
            column_names.append('obj_model')

        df = pd.DataFrame(np.array(data).T, columns=column_names)


        df.to_csv(filename, index=False)




    def reset_mask(self):
        """Reset the spectrum mask by repopulating it with a 1D array of
        boolean 1 values.
        """

        self.mask = np.ones(self.dispersion.shape, dtype=bool)

    def copy(self):
        """Create a new SpecOneD instance, populate it with the values
        from the active spectrum and return it.


        Returns
        -------
        obj:'SpecOneD'
            Returns an new SpecOneD instance populated by the original spectrum.
        """

        return SpecOneD(dispersion=self.dispersion,
                        flux=self.flux,
                        flux_err=self.flux_err,
                        header=self.header,
                        unit=self.unit,
                        mask=self.mask,
                        raw_dispersion=self.raw_dispersion,
                        raw_flux=self.raw_flux,
                        raw_flux_err=self.raw_flux_err,
                        )

    def override_raw(self):
        """ Override the raw_dispersion, raw_flux and raw_flux_err
        variables in the SpecOneD class with the current dispersion, flux and
        flux_err values.
        """

        self.raw_dispersion = self.dispersion
        self.raw_flux = self.flux
        self.flux_err = self.raw_flux_err

    def restore(self):
        """ Override the dispersion, flux and rflux_err
        variables in the SpecOneD class with the raw_dispersion, raw_flux and
        raw_flux_err values.
        """

        self.dispersion = self.raw_dispersion
        self.flux = self.raw_flux
        self.flux_err = self.raw_flux_err
        self.reset_mask()

    def check_units(self, secondary_spectrum):
        """ This method checks if the active spectrum and a second spectrum
        have the same flux units.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum to compare units with.

        Raises
        ------
        ValueError
            Raises an Error when the spectra are in different units.

        """

        if self.unit != secondary_spectrum.unit:
            raise ValueError('Spectra are in different units!')

    def to_log_wavelength(self):
        """ Convert the spectrum into logarithmic wavelength units.

        This method converts the dispersion and flux axis to logarithmic
        wavelength units.

        Raises
        ------
        ValueError
            Raises an error, if the dispersion/flux is arealdy in logarithmic
            wavelength.
        """

        if self.unit == 'f_loglam':
            raise ValueError('Spectrum is already in logarithmic wavelength')

        elif self.unit == 'f_lam':
            self.flux = self.flux * self.dispersion
            self.dispersion = np.log(self.dispersion)

        elif self.unit == 'f_nu':
            self.to_wavelength()
            self.to_log_wavelength()

        self.unit='f_loglam'


    def to_wavelength(self):
        """ Convert the spectrum from flux per frequency to flux per
        wavenlength.

        This method converts the flux from erg/s/cm^2/Hz to
        erg/s/cm^2/Angstroem and the dispersion accordingly from Hz to
        Angstroem.

        Raises
        ------
        ValueError
            Raises an error, if the flux is already in wavelength.
        """

        if self.unit == 'f_lam':
            raise ValueError('Dispersion is arealdy in wavelength')
        elif self.unit == 'f_nu':
            self.flux = self.flux * self.dispersion**2 / (c.value * 1e+10)
            self.dispersion = (c.value * 1e+10) / self.dispersion

            self.flux = np.flip(self.flux, axis=0)
            self.dispersion = np.flip(self.dispersion, axis=0)

        elif self.unit == 'f_loglam':
            self.dispersion = np.exp(self.dispersion)
            self.flux = self.flux / self.dispersion
        else:
            raise ValueError('Spectrum unit not recognized: ', self.unit)

        self.unit = 'f_lam'

    def to_frequency(self):
        """ Convert the spectrum from flux per wavelength to flux per
        frequency.

        This method converts the flux from erg/s/cm^2/Angstroem to
        erg/s/cm^2/Hz and the dispersion accordingly from Angstroem to Hz.

        Raises
        ------
        ValueError
            Raises an error, if the flux is already in frequency.
        """

        if self.unit == 'f_nu':
            raise ValueError('Dispersion is already in frequency.')

        elif self.unit == 'f_lam':
            self.flux = self.flux * self.dispersion**2 / (c.value * 1e+10)
            self.dispersion = (c.value * 1e+10) / self.dispersion

            self.flux = np.flip(self.flux, axis=0)
            self.dispersion = np.flip(self.dispersion, axis=0)

        elif self.unit == 'f_loglam':
            self.to_wavelength()
            self.to_frequency()
        else :
            raise ValueError('Spectrum unit not recognized: ', self.unit)

        self.unit = 'f_nu'

    def check_dispersion_overlap(self, secondary_spectrum):
        """Check the overlap between the active spectrum and the
        supplied secondary spectrum.

        This method determines whether the active spectrum (primary) and the
        supplied spectrum (secondary) have overlap in their dispersions.
        Possible cases include:
        i) The primary spectrum is fully overlapping with the secondary.
        ii) The secondary is fully overlapping with this priamy spectrum, but
        not vice versa.
        iii) and iv) There is partial overlap between the spectra.
        v) There is no overlap between the spectra.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum

        Returns
        -------
        overlap : str
            A string indicating what the dispersion overlap between the spectra
            is according to the cases above.
        overlap_min : float
            The minimum value of the overlap region of the two spectra.
        overlap_max : float
            The maximum value of the overlap region of the two spectra.
        """

        self.check_units(secondary_spectrum)

        spec_min = np.min(self.dispersion)
        spec_max = np.max(self.dispersion)

        secondary_min = np.min(secondary_spectrum.dispersion)
        secondary_max = np.max(secondary_spectrum.dispersion)

        if spec_min >= secondary_min and spec_max <= secondary_max:
            return 'primary', spec_min, spec_max
        elif spec_min < secondary_min and spec_max > secondary_max:
            return 'secondary', secondary_min, secondary_max
        elif spec_min <= secondary_min and secondary_min <= spec_max <= secondary_max:
            return 'partial', secondary_min, spec_max
        elif secondary_max >= spec_min >= secondary_min and spec_max >= secondary_max:
            return 'partial', spec_min, secondary_max
        else:
            return 'none', np.NaN, np.NaN

    def match_dispersions(self, secondary_spectrum, match_secondary=True,
                          force=False, interp_kind='linear'):
        """Match the dispersion of the primary and the supplied secondary
        spectrum.


        Notes
        -----
        TODO: Add flux error handling

        Both, primary and secondary, SpecOneD classes are modified in this
        process. The dispersion match identifies the maximum possible overlap
        in the dispersion direction of both spectra and trims them to that
        range.

        If the primary spectrum overlaps fully with the secondary spectrum the
        dispersion of the secondary will be interpolated to the primary
        dispersion.
        If the secondary spectrum overlaps fully with the primary, the primary
        spectrum will be interpolated on the secondary spectrum resolution, but
        this happens only if 'force==True' and 'match_secondary==False'.
        If there is partial overlap between the spectra and 'force==True'
        the secondary spectrum will be interpolated to match the
        dispersion values of the primary spectrum.
        If there is no overlap a ValueError will be raised.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        match_secondary : boolean
            The boolean indicates whether the secondary will always be matched
            to the primary or whether reverse matching, primary to secondary is
            allowed.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exists.

        Raises
        ------
        ValueError
            A ValueError will be raised if there is no overlap between the
            spectra.

        """

        self.check_units(secondary_spectrum)

        overlap, s_min, s_max = self.check_dispersion_overlap(secondary_spectrum)

        if overlap == 'primary':
            secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)

        elif (overlap == 'secondary' and match_secondary is False and force is
        True):
            self.interpolate(secondary_spectrum.dispersion, kind=interp_kind)

        elif (overlap == 'secondary' and match_secondary is True and force is
        True):
            self.trim_dispersion(limits=[s_min, s_max], mode='wav', inplace=True)
            secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)

        elif overlap == 'partial' and force is True:
            self.trim_dispersion(limits=[s_min, s_max], mode='wav', inplace=True)
            secondary_spectrum.interpolate(self.dispersion, kind=interp_kind)

        elif force is False and (overlap == 'secondary' or overlap == 'partial'):
            raise ValueError('There is overlap between the spectra but force is False.')

        elif overlap == 'none':
            raise ValueError('There is no overlap between the primary and \
                             secondary spectrum.')


    def add(self, secondary_spectrum, copy_header='first', force=True,
            inplace=False, copy_flux_err = 'No'):

        """Add the flux in the primary and secondary spectra.

        Notes
        -----
        TODO: implement correct flux error handling, test it
        Users should be aware that in order to add the flux of the two spectra,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj:`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)
        # flux error needs to be taken care of
        # needs to be tested

        if not np.array_equal(self.dispersion, secondary_spectrum.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            self.match_dispersions(secondary_spectrum, force=force)

        new_flux = self.flux + secondary_spectrum.flux

        if copy_flux_err == 'No' and isinstance(self.flux_err,
                                                np.ndarray) and \
                isinstance(secondary_spectrum.flux_err, np.ndarray):
            new_flux_err = np.sqrt(self.flux_err**2 + secondary_spectrum.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = self.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = secondary_spectrum.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = self.header
        elif copy_header == 'last':
            new_header = secondary_spectrum.header

        if inplace:
            self.flux = new_flux
            self.header = new_header
            self.flux_err = new_flux_err
        else:
            return SpecOneD(dispersion=self.dispersion,
                            flux=new_flux,
                            flux_err=new_flux_err,
                            header=new_header,
                            unit=self.unit,
                            mask=self.mask)

    def subtract(self, secondary_spectrum, copy_header='first', force=True,
                 inplace=False, copy_flux_err = 'No'):
        """Subtract the flux of the secondary spectrum from the primary
        spectrum.

        Notes
        -----
        TODO: implement correct flux error handling, test it
        Users should be aware that in order to subtract the flux,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj:`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)
        # flux error needs to be taken care of
        # needs to be tested
        # check for negative values?

        if not np.array_equal(self.dispersion, secondary_spectrum.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            self.match_dispersions(secondary_spectrum, force=force)

        new_flux = self.flux - secondary_spectrum.flux

        if copy_flux_err == 'No' and isinstance(self.flux_err,
                                                np.ndarray) and \
                isinstance(secondary_spectrum.flux_err, np.ndarray):
            new_flux_err = np.sqrt(self.flux_err**2 + secondary_spectrum.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = self.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = secondary_spectrum.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = self.header
        elif copy_header == 'last':
            new_header = secondary_spectrum.header

        if inplace:
            self.flux = new_flux
            self.header = new_header
        else:
            return SpecOneD(dispersion=self.dispersion,
                            flux=new_flux,
                            flux_err=new_flux_err,
                            header=new_header,
                            unit=self.unit,
                            mask=self.mask)

    def divide(self, secondary_spectrum, copy_header='first', force=True,
               inplace=False, copy_flux_err = 'No'):

        """Divide the flux of primary spectrum by the secondary spectrum.

        Notes
        -----
        TODO: implement correct flux error handling, test it
        Users should be aware that in order to add the flux of the two spectra,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj:`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)

        if not np.array_equal(self.dispersion, secondary_spectrum.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            self.match_dispersions(secondary_spectrum, force=force, match_secondary=False)

            print  (self.dispersion.shape, secondary_spectrum.dispersion.shape)

        new_flux = self.flux / secondary_spectrum.flux

        if copy_flux_err == 'No' and isinstance(self.flux_err,
                                               np.ndarray) and \
                isinstance(secondary_spectrum.flux_err, np.ndarray):
                new_flux_err = np.sqrt( (self.flux_err/ secondary_spectrum.flux)**2  + (new_flux/secondary_spectrum.flux*secondary_spectrum.flux_err)**2 )
        elif copy_flux_err == 'first':
            new_flux_err = self.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = secondary_spectrum.flux_err
        else:
            new_flux_err = None


        if copy_header == 'first':
            new_header = self.header
        elif copy_header == 'last':
            new_header = secondary_spectrum.header

        if inplace:
            self.flux = new_flux
            self.header = new_header
            self.flux_err = new_flux_err
        else:
            return SpecOneD(dispersion=self.dispersion,
                            flux=new_flux,
                            flux_err=new_flux_err,
                            header=new_header,
                            unit=self.unit,
                            mask=self.mask)

    def multiply(self, secondary_spectrum, copy_header='first', force=True,
                 inplace=False, copy_flux_err = 'No'):

        """Multiply the flux of primary spectrum with the secondary spectrum.

        Notes
        -----
        Users should be aware that in order to add the flux of the two spectra,
        the dispersions of the spectra need to be matched, see match_dispersions
        and beware of the caveats of dispersion interpolation.

        Parameters
        ----------
        secondary_spectrum : obj:`SpecOneD`
            Secondary spectrum
        copy_header : 'str'
            A string indicating whether the primary('first') spectrum header or
            the secondary('last') spectrum header should be copied to the
            resulting spectrum.
        force : boolean
            The boolean sets whether the dispersions are matched if only
            partial overlap between the spectral dispersions exist.
        inplace : boolean
            The boolean indicates whether the resulting spectrum will overwrite
            the primary spectrum or whether it will be returned as a new
            spectrum argument by the method.

        Returns
        -------
        obj;`SpecOneD`
            Returns a SpecOneD object in the default case, "inplace==False".
        """

        self.check_units(secondary_spectrum)

        if not np.array_equal(self.dispersion, secondary_spectrum.dispersion):
            print ("Warning: Dispersion does not match.")
            print ("Warning: Flux will be interpolated.")

            self.match_dispersions(secondary_spectrum, force=force)

        new_flux = self.flux * secondary_spectrum.flux

        if copy_flux_err == 'No' and isinstance(self.flux_err,
                                                np.ndarray) and \
                isinstance(secondary_spectrum.flux_err, np.ndarray):
            new_flux_err = np.sqrt(secondary_spectrum.flux**2 * self.flux_err**2 + self.flux**2 * secondary_spectrum.flux_err**2)
        elif copy_flux_err == 'first':
            new_flux_err = self.flux_err
        elif copy_flux_err == 'second':
            new_flux_err = secondary_spectrum.flux_err
        else:
            new_flux_err = None

        if copy_header == 'first':
            new_header = self.header
        elif copy_header == 'last':
            new_header = secondary_spectrum.header

        if inplace:
            self.flux = new_flux
            self.header = new_header
            self.flux_err = new_flux_err
        else:
            return SpecOneD(dispersion=self.dispersion,
                            flux=new_flux,
                            flux_err=new_flux_err,
                            header=new_header,
                            unit=self.unit,
                            mask=self.mask)


    def plot(self, show_flux_err=False, show_raw_flux=False, mask_values=False):

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
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

        elif self.unit =='f_loglam':
            self.ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,(\log\rm{\AA})^{-1}]$', fontsize=15)

        else:
            raise ValueError("Unrecognized units")

        # If a model spectrum exists, print it
        if self.model_spectrum:
            model_flux = self.model_spectrum.eval(self.model_pars, x=self.dispersion)
            self.ax.plot(self.dispersion[mask], model_flux[mask])

        if self.fit_output:
            self.ax.plot(self.dispersion[mask], self.fit_output.best_fit[mask])

        plt.show()


    def pypeit_plot(self, show_flux_err=False, show_raw_flux=False, mask_values=False):

        """Plot the spectrum assuming it is a pypeit spectrum

        """

        if mask_values:
            mask = self.mask
        else:
            mask = np.ones(self.dispersion.shape, dtype=bool)

        self.fig, self.ax = plt.subplots(nrows=1, ncols=1, figsize=(15,7), dpi = 140)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

        # Plot the Spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')

        # Add second axis to plot telluric model
        if hasattr(self, 'telluric'):
            telluric = self.telluric / self.telluric.max() * np.median(
                self.flux[mask]) * 2.5
            self.ax.plot(self.dispersion[mask], telluric, label='Telluric')

        if show_flux_err:
            self.ax.plot(self.dispersion[mask], self.flux_err[mask], 'grey',
                         lw=1, label='Flux Error')
        if show_raw_flux:
            self.ax.plot(self.raw_dispersion[mask], self.raw_flux[mask],
                         'grey', lw=3, label='Raw Flux')

        self.ax.plot(self.dispersion[mask], self.flux[mask], 'k',
                     linewidth=1, label='Flux')

        if self.unit=='f_lam':
            self.ax.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

        elif self.unit =='f_nu':
            self.ax.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

        elif self.unit =='f_loglam':
            self.ax.set_xlabel(r'$\log\rm{Wavelength}\ [\log\rm{\AA}]$', fontsize=15)
            self.ax.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,(\log\rm{\AA})^{-1}]$', fontsize=15)

        else:
            raise ValueError("Unrecognized units")

        # If a model spectrum exists, plot it
        if self.model_spectrum:
            model_flux = self.model_spectrum.eval(self.model_pars, x=self.dispersion)
            self.ax.plot(self.dispersion[mask], model_flux[mask])

        if self.fit_output:
            self.ax.plot(self.dispersion[mask], self.fit_output.best_fit[mask])


        # Add OBJ model if it exists
        if hasattr(self, 'obj_model'):
            self.ax.plot(self.dispersion[mask], self.obj_model, label='Obj '
                                                                      'model')

        self.ax.set_ylim(self.flux[mask].min(), self.flux[mask].max())

        self.ax.legend()
        plt.show()

    def fit_model_spectrum(self, mask_values=True):

        if mask_values:
            self.fit_output = self.model_spectrum.fit(self.flux[self.mask], self.model_pars, x =self.dispersion[self.mask])
        else:
            self.fit_output = self.model_spectrum.fit(self.flux, self.model_pars, x =self.dispersion)

    def calculate_snr(self):

        pass

    def trim_dispersion(self, limits, mode='wav', trim_err=True, inplace=False):

        #change names of modes.... physical, pixel ?

        if mode == "wavelength" or mode == "wav":

            lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
            up_index = np.argmin(np.abs(self.dispersion - limits[1]))

            # Warnings
            if limits[0] < self.dispersion[0]:
                print (self.dispersion[0], limits[0])
                print("WARNING: Lower limit is below the lowest dispersion value. The lower limit is set to the minimum dispersion value.")
            if limits[1] > self.dispersion[-1]:
                print("WARNING: Upper limit is above the highest dispersion value. The upper limit is set to the maximum dispersion value.")



        else:
            # Warnings
            if limits[0] < self.dispersion[0]:
                print("WARNING: Lower limit is below the lowest dispersion value. The lower limit is set to the minimum dispersion value.")
            if limits[1] > self.dispersion[-1]:
                print("WARNING: Upper limit is above the highest dispersion value. The upper limit is set to the maximum dispersion value.")

            lo_index = limits[0]
            up_index = limits[1]

        if inplace:
            self.dispersion = self.dispersion[lo_index:up_index]
            self.flux = self.flux[lo_index:up_index]
            self.mask = self.mask[lo_index:up_index]
            if trim_err:
                if self.flux_err is not None:
                    self.flux_err = self.flux_err[lo_index:up_index]
        else:
            spec = self.copy()
            spec.dispersion = spec.dispersion[lo_index:up_index]
            spec.flux = spec.flux[lo_index:up_index]
            spec.mask = spec.mask[lo_index:up_index]
            if trim_err:
                if spec.flux_err is not None:
                    spec.flux_err = spec.flux_err[lo_index:up_index]

            return spec


    def interpolate(self, new_dispersion, kind='linear', fill_value='const'):

        """Interpolate flux to new dispersion axis

        Parameters
        ----------
        new_dispersion : ndarray
            1D array with the new dispersion axis
        kind : str
            String that indicates the interpolation function
        fill_value : str
            A string indicating whether values outside the dispersion range
            will be extrapolated ('extrapolate') or filled with a constant
            value ('const') based on the median of the 10 values at the edge.
        """

        if fill_value=='extrapolate':
            f = sp.interpolate.interp1d(self.dispersion, self.flux, kind=kind, fill_value='extrapolate')
            if isinstance(self.flux_err, np.ndarray):
                f_err = sp.interpolate.interp1d(self.dispersion, self.flux_err, kind=kind, fill_value='extrapolate')
            print ('Warning: Values outside the original dispersion range will be extrapolated!')
        elif fill_value == 'const':
            fill_lo = np.median(self.flux[0:10])
            fill_hi = np.median(self.flux[-11:-1])
            f = sp.interpolate.interp1d(self.dispersion, self.flux, kind=kind,
                                        bounds_error= False,
                                        fill_value=(fill_lo, fill_hi))
            if isinstance(self.flux_err, np.ndarray):
                fill_lo_err = np.median(self.flux_err[0:10])
                fill_hi_err = np.median(self.flux_err[-11:-1])
                f_err = sp.interpolate.interp1d(self.dispersion, self.flux_err, kind=kind, bounds_error= False,
                                        fill_value=(fill_lo_err, fill_hi_err))
        else:
            f = sp.interpolate.interp1d(self.dispersion, self.flux, kind=kind)
            if isinstance(self.flux_err, np.ndarray):
                f_err = sp.interpolate.interp1d(self.dispersion, self.flux_err, kind=kind)

        self.dispersion = new_dispersion
        self.reset_mask()
        self.flux = f(self.dispersion)
        if isinstance(self.flux_err, np.ndarray):
            self.flux_err = f_err(self.dispersion)

    def smooth(self, width, kernel="boxcar", inplace=False):

        if kernel == "boxcar" or kernel == "Boxcar":
            kernel = Box1DKernel(width)
        elif kernel == "gaussian" or kernel == "Gaussian":
            kernel = Gaussian1DKernel(width)

        if inplace:
            self.flux = convolve(self.flux, kernel)
        else:
            flux = convolve(self.flux, kernel)
            return SpecOneD(dispersion=self.dispersion,
                            flux=flux,
                            flux_err=self.flux_err,
                            header=self.header,
                            unit = self.unit)

    def convolve_loglam(self, fwhm, inplace=False):
        """Convolve the spectrum in loglam space with a kernel function of specified width"""

        spec = self.copy()

        stddev = fwhm / (2 * np.sqrt(2 * np.log(2)))

        # Step 1 convert spectrum to logarithmic wavelength (velocity space)
        spec.to_log_wavelength()
        # Interpolate to linear scale in logarithmic wavelength
        new_disp = np.linspace(min(spec.dispersion), max(spec.dispersion),num = len(spec.dispersion))
        spec.interpolate(new_disp)

        # Step 2 convolve spectrum with function
        cen = (max(new_disp)-min(new_disp))/2. + min(new_disp)
        y = gaussian(new_disp, 1.0, cen, stddev/300000, 0)
        # hacked the normalization here... did not fully understand why this happens
        conv = np.convolve(spec.flux, y, mode='same') / (len(new_disp)/2.)

        spec.flux = conv

        # Step 3 convert back to original space
        spec.to_wavelength()
        spec.interpolate(self.dispersion)

        if inplace:
            self.flux = spec.flux
        else:
            return SpecOneD(dispersion=spec.dispersion,
                            flux=spec.flux,
                            flux_err=spec.flux_err,
                            header=spec.header,
                            unit = spec.unit)


    def redden(self, a_v, r_v, extinction_law='ccm89', inplace=False):

        if self.unit != 'f_lam':
            raise ValueError('Dispersion units must be in wavelength (Angstroem)')

        if extinction_law == 'ccm89':
            extinction = ext.ccm89(self.dispersion, a_v, r_v)
        elif extinction_law == 'odonnel94':
            extinction = ext.odonnel94(self.dispersion, a_v, r_v)
        elif extinction_law == 'calzetti00':
            extinction = ext.calzetti00(self.dispersion, a_v, r_v)
        elif extinction_law == 'fitzpatrick99':
            extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
        elif extinction_law == 'fm07':
            print('Warning: For Fitzpatrick & Massa 2007 R_V=3.1')
            extinction = ext.fm07(self.dispersion, a_v)
        else:
            raise ValueError('Specified Extinction Law not recognized')

        if inplace:
            self.flux = self.flux * 10.0**(-0.4*extinction)
        else:
            flux = self.flux * 10.0**(-0.4*extinction)
            return SpecOneD(dispersion=self.dispersion,
                            flux=flux,
                            flux_err=self.flux_err,
                            header=self.header,
                            unit = self.unit)

    def deredden(self, a_v, r_v, extinction_law='ccm89', inplace=False):

        if self.unit != 'f_lam':
            raise ValueError('Dispersion units must be in wavelength (Angstroem)')

        if extinction_law == 'ccm89':
            extinction = ext.ccm89(self.dispersion, a_v, r_v)
        elif extinction_law == 'odonnel94':
            extinction = ext.odonnel94(self.dispersion, a_v, r_v)
        elif extinction_law == 'calzetti00':
            extinction = ext.calzetti00(self.dispersion, a_v, r_v)
        elif extinction_law == 'fitzpatrick99':
            extinction = ext.fitzpatrick99(self.dispersion, a_v, r_v)
        elif extinction_law == 'fm07':
            print('Warning: For Fitzpatrick & Massa 2007 R_V=3.1')
            extinction = ext.fm07(self.dispersion, a_v)
        else:
            raise ValueError('Specified Extinction Law not recognized')

        if inplace:
            self.flux = self.flux * 10.0**(0.4*extinction)
        else:
            flux = self.flux * 10.0**(0.4*extinction)
            return SpecOneD(dispersion=self.dispersion,
                            flux=flux,
                            flux_err=self.flux_err,
                            header=self.header,
                            unit = self.unit)


    def calculate_passband_flux(self, passband, force=False):

        spec = self.copy()

        if spec.unit == 'f_lam':
            spec.to_frequency()
        elif spec.unit != 'f_nu':
            raise ValueError('Spectrum units must be f_lam or f_nu')
        if passband.unit == 'f_lam':
            passband.to_frequency()
        elif passband.unit != 'f_nu':
            raise ValueError('PassBand units must be f_lam or f_nu')


        overlap, disp_min, disp_max = passband.check_dispersion_overlap(spec)

        if not force:
            if overlap != 'primary':
                raise ValueError('The spectrum does not fill the passband')
        else:
            print(
                "Warning: Force was set to TRUE. The spectrum might not fully fill the passband!")

        passband.match_dispersions(spec, force=force)
        spec.flux = passband.flux * spec.flux
        total_flux = np.trapz(spec.flux, spec.dispersion)

        if total_flux <= 0.0:
            raise ValueError('Integrated flux is <= 0')
        if np.isnan(total_flux):
            raise ValueError('Integrated flux is NaN')
        if np.isinf(total_flux):
            raise ValueError('Integrated flux is infinite')

        return total_flux

    def calculate_passband_magnitude(self, passband, mag_system='AB',
                                      force=False):
        # This function is written for passbands in quantum efficiency
        # Therefore, the (h*nu)^-1 term is not included in the integral

        spec = self.copy()

        if mag_system == 'AB':
            if spec.unit == 'f_lam':
                spec.to_frequency()
            elif spec.unit != 'f_nu':
                raise ValueError('Spectrum units must be f_lam or f_nu')
            if passband.unit == 'f_lam':
                passband.to_frequency()
            elif passband.unit != 'f_nu':
                raise ValueError('PassBand units must be f_lam or f_nu')
        else:
            raise NotImplementedError('Only AB magnitudes are currently implemented')

        overlap, disp_min, disp_max = passband.check_dispersion_overlap(spec)

        if not force:
            if overlap != 'primary':
                raise ValueError('The spectrum does not fill the passband')
        else:
            print ("Warning: Force was set to TRUE. The spectrum might not fully fill the passband!")

        passband.match_dispersions(spec, force=force)

        spec.flux = passband.flux * spec.flux

        total_flux = np.trapz(spec.flux , spec.dispersion)

        if total_flux <= 0.0:
            raise ValueError('Integrated flux is <= 0')
        if np.isnan(total_flux):
            raise ValueError('Integrated flux is NaN')
        if np.isinf(total_flux):
            raise ValueError('Integrated flux is infinite')

        flat = FlatSpectrum(spec.dispersion, unit='f_nu')

        passband_flux = np.trapz(flat.flux * passband.flux ,
                                 flat.dispersion)

        ratio = total_flux / passband_flux

        return -2.5 * np.log10(ratio)

    def renormalize_by_magnitude(self, magnitude, passband, mag_system='AB',
                                 force=False, inplace=False):

        spec_mag = self.calculate_passband_magnitude(passband,
                                                     mag_system=mag_system,
                                                     force=force)

        dmag = magnitude - spec_mag

        if inplace:
            self.to_frequency
            self.flux_scale_factor = 10**(-0.4*dmag)
            self.flux = self.flux * 10**(-0.4*dmag)
            self.to_wavelength
        else:
            spec = self.copy()
            spec.to_frequency
            spec.flux_scale_factor = 10**(-0.4*dmag)
            spec.flux = spec.flux * 10**(-0.4*dmag)
            spec.to_wavelength

            return spec



    def renormalize_by_spectrum(self, spectrum, dispersion='match', trim_mode='wav', inplace=False):


        """ Match the flux of the active spectrum to the "spectrum" given to
        the function.

        Note:
        Both spectra will be copied to dummy variables:
        spec = active spectrum
        spec2 = supplied spectrum
        """

        spec = self.copy()
        spec2 = spectrum.copy()



        if dispersion == "match":
            spec.match_dispersions(spec2, match_secondary=False,
                                  force=True, interp_kind='linear')
        elif isinstance(dispersion,(list,)):
            spec.trim_dispersion(dispersion, mode=trim_mode,inplace=True)
            spec2.trim_dispersion(dispersion, mode=trim_mode,inplace=True)
        else:
            print("Spectra will be normalized but dispersion ranges do not necessarily match!")



        average_self_flux = np.trapz(spec.flux, spec.dispersion)

        average_spec_flux = np.trapz(spec2.flux, spec2.dispersion)

        self.scale = (average_spec_flux/average_self_flux)

        if inplace:
            self.flux_scale_factor = average_spec_flux/average_self_flux
            self.flux = self.flux * (average_spec_flux/average_self_flux)
        else:
            spec = self.copy()
            spec.flux_scale_factor = average_spec_flux/average_self_flux
            flux = self.flux * (average_spec_flux/average_self_flux)
            spec.scale = self.scale
            spec.flux = flux

            return spec

    def redshift(self, z, inplace=False):

        if inplace:
            self.dispersion = self.dispersion * (1.+z)
        else:
            spec = self.copy()
            spec.dispersion = spec.dispersion * (1.+z)

            return spec

    def sigmaclip_flux(self, low=3, up=3, binsize=120, niter=5, inplace=False):

        hbinsize = int(binsize/2)

        flux = self.flux
        dispersion = self.dispersion


        mask_index = np.arange(dispersion.shape[0])

        # loop over sigma-clipping iterations
        for jdx in range(niter):

            n_mean = np.zeros(flux.shape[0])
            n_std = np.zeros(flux.shape[0])

            # calculating mean and std arrays
            for idx in range(len(flux[:-binsize])):

                # flux subset
                f_bin = flux[idx:idx+binsize]

                # calculate mean and std
                mean = np.median(f_bin)
                std = np.std(f_bin)

                # set array value
                n_mean[idx+hbinsize] = mean
                n_std[idx+hbinsize] = std

            # fill zeros at end and beginning
            # with first and last values
            n_mean[:hbinsize] = n_mean[hbinsize]
            n_mean[-hbinsize:] = n_mean[-hbinsize-1]
            n_std[:hbinsize] = n_std[hbinsize]
            n_std[-hbinsize:] = n_std[-hbinsize-1]

            # create index array with included pixels ("True" values)
            mask = (flux-n_mean < n_std*up) & (flux-n_mean > -n_std*low)
            mask_index = mask_index[mask]

            # mask the flux for the next iteration
            flux = flux[mask]

        mask = np.zeros(len(self.mask), dtype='bool')

        # mask = self.mask
        mask[:] = False
        mask[mask_index] = True
        mask = mask * self.mask

        if inplace:
            self.mask = mask
        else:
            spec = self.copy()
            spec.mask = mask

            return spec

    def fit_polynomial(self, func= 'legendre', order=3, inplace=False):

        if func == 'legendre':
            poly = Legendre.fit(self.dispersion[self.mask], self.flux[self.mask], deg=order)
        elif func == 'chebyshev':
            poly = Chebyshev.fit(self.dispersion[self.mask], self.flux[self.mask], deg=order)
        elif func == 'polynomial':
            poly = Polynomial.fit(self.dispersion[self.mask], self.flux[self.mask], deg=order)
        else:
            raise ValueError("Polynomial fitting function not specified")

        if inplace:
            self.fit_dispersion = self.dispersion
            self.fit_flux = poly(self.dispersion)
        else:
            spec = self.copy()
            spec.fit_dispersion = self.dispersion
            spec.fit_flux = poly(self.dispersion)

            return spec

    def mask_between(self, limits, inplace=False):

        lo_index = np.argmin(np.abs(self.dispersion - limits[0]))
        up_index = np.argmin(np.abs(self.dispersion - limits[1]))

        self.mask[lo_index:up_index] = 0

        if inplace:
            self.mask[lo_index:up_index] = 0
        else:
            spec = self.copy()
            spec.mask[lo_index:up_index] = 0

            return spec

    def resample(self, new_dispersion, inplace=False):
        """
        Function for resampling spectra (and optionally associated
        uncertainties) onto a new wavelength basis.

        TODO TEST IT AND THEN CHANGE DEFAULT TO RESAMPLE INSTEAD OF INTERPOLATE

        This code is copied from
        https: // github.com / ACCarnall / SpectRes
        by Adam Carnall - damc@roe.ac.uk

        and adapted by Jan-Torge Schindler for functionality within
        the SpecOneD class

        Parameters
        ----------
        new_dispersion : numpy.ndarray
            Array containing the new wavelength sampling desired for the
            spectrum or spectra.

        inplace : Boolean
            Boolean to indicate whether the results overwrite the SpecOneD
            object or a new SpecOneD object is created and returned.

        Returns
        -------
        spec/self : SpecOneD
            The function returns a SpecOneD class object with the new
            dispersion and the resampled fluxes (and flux errors).
        """

        # Mapping of the SpecOneD object variables to the function
        # variables

        old_spec_wavs = self.dispersion
        spec_fluxes = self.flux
        if self.flux_err is not None:
            spec_errs = self.flux_err

        new_spec_wavs = new_dispersion

        # Arrays of left-hand sides and widths for the old and new bins
        spec_widths = np.zeros(old_spec_wavs.shape[0])
        spec_lhs = np.zeros(old_spec_wavs.shape[0])
        spec_lhs[0] = old_spec_wavs[0]
        spec_lhs[0] -= (old_spec_wavs[1] - old_spec_wavs[0]) / 2
        spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
        spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1]) / 2
        spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

        filter_lhs = np.zeros(new_spec_wavs.shape[0] + 1)
        filter_widths = np.zeros(new_spec_wavs.shape[0])
        filter_lhs[0] = new_spec_wavs[0]
        filter_lhs[0] -= (new_spec_wavs[1] - new_spec_wavs[0]) / 2
        filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
        filter_lhs[-1] = new_spec_wavs[-1]
        filter_lhs[-1] += (new_spec_wavs[-1] - new_spec_wavs[-2]) / 2
        filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1]) / 2
        filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

        if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
            raise ValueError("spectres: The new wavelengths specified must fall"
                             "within the range of the old wavelength values:",
                             filter_lhs[0], spec_lhs[0], filter_lhs[-1],
                             spec_lhs[-1])

        # Generate output arrays to be populated
        res_fluxes = np.zeros(spec_fluxes[..., 0].shape + new_spec_wavs.shape)

        if spec_errs is not None:
            if spec_errs.shape != spec_fluxes.shape:
                raise ValueError(
                    "If specified, spec_errs must be the same shape"
                    "as spec_fluxes.")
            else:
                res_fluxerrs = np.copy(res_fluxes)

        start = 0
        stop = 0

        # Calculate new flux and uncertainty values, loop over new bins
        for j in range(new_spec_wavs.shape[0]):

            # Find first old bin which is partially covered by the new bin
            while spec_lhs[start + 1] <= filter_lhs[j]:
                start += 1

            # Find last old bin which is partially covered by the new bin
            while spec_lhs[stop + 1] < filter_lhs[j + 1]:
                stop += 1

            # If new bin is fully within one old bin these are the same
            if stop == start:

                res_fluxes[..., j] = spec_fluxes[..., start]
                if spec_errs is not None:
                    res_fluxerrs[..., j] = spec_errs[..., start]

            # Otherwise multiply the first and last old bin widths by P_ij
            else:

                start_factor = ((spec_lhs[start + 1] - filter_lhs[j])
                                / (spec_lhs[start + 1] - spec_lhs[start]))

                end_factor = ((filter_lhs[j + 1] - spec_lhs[stop])
                              / (spec_lhs[stop + 1] - spec_lhs[stop]))

                spec_widths[start] *= start_factor
                spec_widths[stop] *= end_factor

                # Populate res_fluxes spectrum and uncertainty arrays
                f_widths = spec_widths[start:stop + 1] * spec_fluxes[...,
                                                         start:stop + 1]
                res_fluxes[..., j] = np.sum(f_widths, axis=-1)
                res_fluxes[..., j] /= np.sum(spec_widths[start:stop + 1])

                if spec_errs is not None:
                    e_wid = spec_widths[start:stop + 1] * spec_errs[...,
                                                          start:stop + 1]

                    res_fluxerrs[..., j] = np.sqrt(np.sum(e_wid ** 2, axis=-1))
                    res_fluxerrs[..., j] /= np.sum(spec_widths[start:stop + 1])

                # Put back the old bin widths to their initial values for
                # later use
                spec_widths[start] /= start_factor
                spec_widths[stop] /= end_factor

        if inplace:

            self.dispersion = new_dispersion
            self.flux = res_fluxes
            if spec_errs is not None:
                self.flux_err = res_fluxerrs

            self.reset_mask()

        else:

            spec = self.copy()

            spec.dispersion = new_dispersion
            spec.flux = res_fluxes
            if spec_errs is not None:
                spec.flux_err = res_fluxerrs

            spec.reset_mask()

            return spec


    def clean_outliers(self):

        # mask outliers, interpolate to fill flux values that were masked

        pass


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

def combine_spectra(filenames, method='average', file_format='fits'):

    s_list = []

    for filename in filenames:
        spec = SpecOneD()
        if file_format == 'fits':
            spec.read_from_fits(filename)
            print(spec)
        else:
            raise NotImplementedError('File format not understood')

        s_list.append(spec)

    # TODO Test if all spectra are in same unit, if not convert
    print(s_list)
    # Resample all spectra onto slightly reduced dispersion of first spectrum
    disp = s_list[0].dispersion[5:-5]
    for spec in s_list:
        spec.resample(disp, inplace=True)

    comb_flux = np.zeros(len(disp))
    comb_fluxerr = np.zeros(len(disp))

    N = float(len(filenames))

    if method == 'average':
        for spec in s_list:
            comb_flux += spec.flux
            comb_fluxerr += spec.flux_err**2/N**2

        comb_flux = comb_flux / N
        comb_fluxerr = np.sqrt(comb_fluxerr)

        comb_spec = SpecOneD(dispersion=disp, flux=comb_flux,
                             flux_err=comb_fluxerr, unit='f_lam',
                             )
        return comb_spec

    else:
        raise NotImplementedError('Selected method for combining spectra is '
                                  'not implemented. Implemented methods: '
                                  'average')






def comparison_plot(spectrum_a, spectrum_b, spectrum_result, show_flux_err=True):

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15,7), dpi=140)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)

    ax1.plot(spectrum_a.dispersion, spectrum_a.flux, color='k')
    ax1.plot(spectrum_b.dispersion, spectrum_b.flux, color='r')

    ax2.plot(spectrum_result.dispersion, spectrum_result.flux, color='k')

    if show_flux_err:
        ax1.plot(spectrum_a.dispersion, spectrum_a.flux_err, 'grey', lw=1)
        ax1.plot(spectrum_b.dispersion, spectrum_b.flux_err, 'grey', lw=1)
        ax2.plot(spectrum_result.dispersion, spectrum_result.flux_err, 'grey', lw=1)

    if spectrum_result.unit=='f_lam':
        ax2.set_xlabel(r'$\rm{Wavelength}\ [\rm{\AA}]$', fontsize=15)
        ax1.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)
        ax2.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{\AA}^{-1}]$', fontsize=15)

    elif spectrum_result.unit =='f_nu':
        ax2.set_xlabel(r'$\rm{Frequency}\ [\rm{Hz}]$', fontsize=15)
        ax1.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)
        ax2.set_ylabel(r'$\rm{Flux}\ f_{\nu}\ [\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{2}\,\rm{Hz}^{-1}]$', fontsize=15)

    else :
        raise ValueError("Unrecognized units")

    plt.show()

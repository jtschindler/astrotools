


import os
import sys
import time
import glob
import json
import numpy as np
import scipy as sp
import pandas as pd
import astropy.constants as const
from astropy.modeling.blackbody import blackbody_lambda
from astrotools.speconed import speconed as sod
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from lmfit.model import save_model, load_model, save_modelresult, load_modelresult
from lmfit.models import ExponentialModel, GaussianModel, LinearModel, VoigtModel

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel, QHBoxLayout, QLineEdit, QCheckBox, QFileDialog, QComboBox,  QScrollArea, QGroupBox
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib.transforms as mtransforms



from .speconed import datadir
# datadir = os.path.split(__file__)[0]
# datadir = os.path.split(datadir)[0] + '/data/'

c_km_s = const.c.to('km/s').value


# ------------------------------------------------------------------------------
# Functions for Models
# ------------------------------------------------------------------------------

def gaussian(x, amp, cen, sigma, shift):
    """ Calculate 1-D Gaussian

    :param x: ndarray
         x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param sigma: float
        Standard deviation of the Gaussian
    :param shift: float
        x-Axis shift of the Gaussian
    :return: ndarray

    """

    central = cen + shift
    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


def gaussian_fwhm(x, amp, cen, fwhm, shift):
    """ Calculate 1-D Gaussian using fwhm instead sigma

    :param x: ndarray
        x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param fwhm: float
        Full Width at Half Maximum (FWHM) of the Gaussian
    :param shift: float
        x-Axis shift of the Gaussian
    :return: ndarray
    """

    cen = cen + shift
    sigma = fwhm / np.sqrt(8*np.log(2))
    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-cen)**2 / (2*sigma**2))


def gaussian_fwhm_km_s(x, amp, cen, fwhm_km_s, shift_km_s):
    """ Calculate 1-D Gaussian using fwhm for in km/s instead sigma

    :param x: ndarray
        x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param fwhm_km_s: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s: float
        x-Axis shift of the Gaussian in km/s
    :return: ndarray
    """

    delta_cen = shift_km_s / c_km_s * cen
    central = cen + delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


def gaussian_fwhm_km_s_z(x, z, amp, cen, fwhm_km_s, shift_km_s):
    """ Calculate 1-D Gaussian using fwhm for in km/s instead sigma

    :param x: ndarray
        x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param fwhm_km_s: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s: float
        x-Axis shift of the Gaussian in km/s
    :return: ndarray
    """

    cen = cen * (1+z)

    delta_cen = shift_km_s / c_km_s * cen
    central = cen + delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


def gaussian_fwhm_z(x, z, amp, cen, fwhm_km_s, shift_z):
    """ Calculate 1-D Gaussian using fwhm for in km/s instead sigma

    :param x: ndarray
        x-Axis of the Gaussian
    :param amp: float
        Amplitude of the Gaussian
    :param cen: float
        Central x-value of the Gaussian
    :param fwhm_km_s: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s: float
        x-Axis shift of the Gaussian in km/s
    :return: ndarray
    """

    central = cen * (1+z+shift_z)


    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8*np.log(2))

    return (amp / (np.sqrt(2*np.pi) * sigma)) * np.exp(-(x-central)**2 / (2*sigma**2))


def power_law(x, amp, slope):
    """ Power law

    Parameters:
    -----------
    :param x: ndarray
        x-Axis of the power law
    :param amp: float
        Amplitude of the power law
    :param slope: float
        Slope of the power law

    Returns:
    --------
    :return: ndarray
    """

    return amp*(x**slope)

def power_law_at_2500A(x, amp, slope, z):
    """ Power law anchored at 2500 (Angstroem)

    Parameters:
    -----------
    :param x: ndarray
        x-Axis of the power law
    :param amp: float
        Amplitude of the power law anchored at 2500 (Angstroem)
    :param slope: float
        Slope of the power law

    Returns:
    --------
    :return: ndarray
    """

    return amp * (x / (2500. * (z+1.)))**slope

def planck_function(x, Te):



    planck_function = (2 * h * c ** 2) / x ** 5 / (
                np.exp((h * c) / (x * kB * T)) - 1)

    return planck_function

def power_law_at_2500A_plus_BC(x, amp, slope, z, T_e, tau_BE, lambda_BE):
    """
    Power law anchored at 2500 (Angstroem) plus a Balmer continuum model with
    a fixed flux of 30% of the power law flux at 3645A.
    :param x:
    :param amp:
    :param slope:
    :param z:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """



    pl_flux =  amp * (x / (2500. * (z + 1.))) ** slope

    pl_flux_at_BE = amp * ((lambda_BE * (z + 1.)) / (2500. * (z + 1.))) ** slope


    x = x / (1. + z)

    bc_flux_at_lambda_BE = blackbody_lambda(lambda_BE, T_e).value * (
                1. - np.exp(-tau_BE))

    bc_flux = 0.3 * pl_flux_at_BE / bc_flux_at_lambda_BE * \
              blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    bc_flux[x >= lambda_BE] = bc_flux[x >= lambda_BE] * 0

    return pl_flux + bc_flux


def power_law_at_2500A_plus_flexible_BC(x, amp, slope, z, f, T_e, tau_BE,
                                       lambda_BE):
    """
    Power law anchored at 2500 (Angstroem) plus a Balmer continuum model with
    a fixed flux of 30% of the power law flux at 3645A.
    :param x:
    :param amp:
    :param slope:
    :param z:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """



    pl_flux =  amp * (x / (2500. * (z + 1.))) ** slope

    pl_flux_at_BE = amp * ((lambda_BE * (z + 1.)) / (2500. * (z + 1.))) ** slope


    x = x / (1. + z)

    bc_flux_at_lambda_BE = blackbody_lambda(lambda_BE, T_e).value * (
                1. - np.exp(-tau_BE))

    bc_flux = f * pl_flux_at_BE / bc_flux_at_lambda_BE * \
              blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    bc_flux[x >= lambda_BE] = bc_flux[x >= lambda_BE] * 0

    return pl_flux + bc_flux


def power_law_at_2500A_plus_manual_BC(x, amp, slope, z, amp_BE, T_e, tau_BE,
                                       lambda_BE):
    """
    Power law anchored at 2500 (Angstroem) plus a Balmer continuum model with
    a fixed flux of 30% of the power law flux at 3645A.
    :param x:
    :param amp:
    :param slope:
    :param z:
    :param T_e:
    :param tau_BE:
    :param lambda_BE:
    :return:
    """



    pl_flux =  amp * (x / (2500. * (z + 1.))) ** slope

    # pl_flux_at_BE = amp * ((lambda_BE * (z + 1.)) / (2500. * (z + 1.))) ** slope


    x = x / (1. + z)

    F_BC0 = amp_BE /(blackbody_lambda(lambda_BE, T_e).value * (
                1. - np.exp(-tau_BE)))

    bc_flux = F_BC0 * \
              blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    bc_flux[x >= lambda_BE] = bc_flux[x >= lambda_BE] * 0

    return pl_flux + bc_flux


# ------------------------------------------------------------------------------
# Basic Models
# ------------------------------------------------------------------------------

def emission_line_model(amp, cen, wid, shift, unit_type, expr_list = None,
                        fit_central=True, fit_width=True, fit_shift=True,
                        prefix=None, parameters=None, redsh=None):
    """ Create a lmfit model for an emission line for 1D astronomical spectra.

    Parameters
    ----------
    :param amp : float
        amplitude of the emission line model
    :param cen : float
        Central wavelength of the emission line
    :param wid : float
        Width of the emission line specified in width_type
    :param unit_type : string
        A string defining the type of the emission line width  and shift input,
        possible types are
        "sigma" : standard deviation of a gaussian and shift in wavelength
        units (Angstroem)
        "fwhm" : full width at half maximum of a gaussian and shift in
        wavelength units (Angstroem),
        "fwhm_km_s" : full width at half maximum of a gaussian and shift in
        km/s
    :param shift : float
        Wavelength shift of the emission line around the central wavelength
    :param fit_central : boolean
        Boolean to determine whether the central wavelength is fit or not
    :param fit_width : boolean
        Boolean to determine whether the width is fit or not
    fit_shift : boolean
        Boolean to determine whether the wavelength shift is fit or not
    prefix : string
        Prefix, which is added to the model parameters

    Raises
    ------
    :raises ValueError

    Returns
    -------
    :return elmodel : lmfit.Model()
        Returns a lmfit Model() object
    """

    if parameters :
        params = parameters
    else:
        params = Parameters()


    params.add(prefix+'amp', value=amp, vary=True)

    if unit_type in ["fwhm_km_s_z", 'fwhm_z']:
        params.add(prefix + 'z', value=redsh, min=redsh * 0.9,
                   max=max(redsh * 1.1, 1),
                   vary=True)

    if isinstance(cen, float) or isinstance(cen, int):
        params.add(prefix+'cen', value=cen, vary=fit_central)
    else:
        raise ValueError('The central wavelength needs to be a float or a'
                         ' string (for conditional fitting expressions)')

    if unit_type == "sigma":

        params.add(prefix+'sigma', value=wid, vary=fit_width)

        if isinstance(shift, float) or isinstance(shift, int):
            params.add(prefix+'shift', value=shift, vary=fit_shift)
        else:
            raise ValueError('The shift needs to be a float or a string'
                             ' (for conditional fitting expressions)')

        elmodel = Model(gaussian, prefix=prefix)

    elif unit_type == "fwhm":

        params.add(prefix+'fwhm', value=wid, vary=fit_width)

        if isinstance(shift, float) or isinstance(shift, int):
            params.add(prefix+'shift', value=shift, vary=fit_shift)
        else:
            raise ValueError('The shift needs to be a float or a string'
                             ' (for conditional fitting expressions)')


        elmodel = Model(gaussian_fwhm, prefix=prefix)

    elif unit_type == "fwhm_km_s":

        params.add(prefix+'fwhm_km_s', value=wid, vary=fit_width)

        if isinstance(shift, float) or isinstance(shift, int):
            params.add(prefix+'shift_km_s', value=shift, vary=fit_shift)
        else:
            raise ValueError('The shift needs to be a float or a string'
                             ' (for conditional fitting expressions)')

        elmodel = Model(gaussian_fwhm_km_s, prefix=prefix)


    elif unit_type == "fwhm_km_s_z":

        params.add(prefix+'fwhm_km_s', value=wid, vary=fit_width)

        if isinstance(shift, float) or isinstance(shift, int):
            params.add(prefix+'shift_km_s', value=shift, vary=fit_shift)
        else:
            raise ValueError('The shift needs to be a float or a string'
                             ' (for conditional fitting expressions)')

        elmodel = Model(gaussian_fwhm_km_s_z, prefix=prefix)

    elif unit_type == "fwhm_z":

        params.add(prefix + 'fwhm_km_s', value=wid, vary=fit_width)

        if isinstance(shift, float) or isinstance(shift, int):
            params.add(prefix + 'shift_z', value=shift, vary=fit_shift)
        else:
            raise ValueError('The shift needs to be a float or a string'
                             ' (for conditional fitting expressions)')

        elmodel = Model(gaussian_fwhm_z, prefix=prefix)

    else:
        raise ValueError('The unit type was not regonized. It needs to be'
                         ' either "sigma", "fwhm", or "fwhm_km_s".')

    return params, elmodel


def template_model(x, amp, z, fwhm, templ_disp=None, templ_flux=None):

    template_spec = sod.SpecOneD(dispersion=templ_disp, flux=templ_flux,
                                 unit='f_lam')
    # artifical broadening
    spec = template_spec.convolve_loglam(fwhm)
    # shift in redshift
    spec = spec.redshift(z)
    # return interpolation function
    f = sp.interpolate.interp1d(spec.dispersion, spec.flux, kind='linear',
                                bounds_error=False, fill_value=(0, 0))

    return f(x)*amp

def template_model_new(x, amp, z, fwhm, intr_fwhm, templ_disp=None,
                   templ_flux=None):

    template_spec = sod.SpecOneD(dispersion=templ_disp, flux=templ_flux,
                                 unit='f_lam')
    # artifical broadening
    convol_fwhm = np.sqrt(fwhm**2-intr_fwhm**2)
    spec = template_spec.convolve_loglam(convol_fwhm)
    # shift in redshift
    spec = spec.redshift(z)
    # return interpolation function
    f = sp.interpolate.interp1d(spec.dispersion, spec.flux, kind='linear',
                                bounds_error=False, fill_value=(0, 0))

    return f(x)*amp

def load_template_model(template_filename=None, fwhm=None, redshift=None,
                        prefix=None, flux_2500=None, wav_limits=None,
                        norm_wavelength=None):

    templ_params = Parameters()
    templ_params.add('z', value=redshift, min=0, max=1089)

    if redshift is None:
        redshift = 1.0
    if fwhm is None:
        fwhm = 900

    template = np.genfromtxt(datadir+'iron_templates/'+template_filename)

    if wav_limits is not None:
        wav_min = wav_limits[0]
        wav_max = wav_limits[1]

        idx_min = np.argmin(np.abs(template[:, 0] - wav_min))
        idx_max = np.argmin(np.abs(template[:, 0] - wav_max))

        templ_model = Model(template_model,
                            templ_disp=template[idx_min:idx_max, 0],
                            templ_flux=template[idx_min:idx_max, 1],
                            prefix=prefix)

    else:
        templ_model = Model(template_model,
                            templ_disp=template[:, 0],
                            templ_flux=template[:, 1],
                            prefix=prefix)



    templ_params.add(prefix +'z', value=redshift, min=0, max=1089)
    templ_params.add(prefix+'fwhm', value=fwhm)
    # Set amplitude to 1 initially and then calculate first  best guess flux
    # below
    templ_params.add(prefix+'amp', value=1.0e-4, min=1.0e-10, max=1.0)

    # normalize the template if possible
    if flux_2500 is not None:
        print("FLUX2500", flux_2500)
        if norm_wavelength is None:
            templ_params[prefix + 'amp'].set(value=flux_2500)
        else:
            model_flux = templ_model.eval(templ_params,
                                          x=norm_wavelength*(1.+redshift))
            print("Model_flux", model_flux)
            new_amp = flux_2500 / model_flux*0.5
            templ_params[prefix+'amp'].set(value=new_amp)

    else:
        templ_params[prefix + 'amp'].set(value=1.0e-15)

    return templ_model, templ_params

def load_template_model_new(template_filename=None, fwhm=None, redshift=None,
                        prefix=None, flux_2500=None, wav_limits=None,
                        norm_wavelength=None, intr_fwhm=900):

    templ_params = Parameters()
    templ_params.add('z', value=redshift, min=0, max=1089)

    if redshift is None:
        redshift = 1.0
    if fwhm is None:
        fwhm = 900

    template = np.genfromtxt(datadir+'iron_templates/'+template_filename)

    if wav_limits is not None:
        wav_min = wav_limits[0]
        wav_max = wav_limits[1]

        idx_min = np.argmin(np.abs(template[:, 0] - wav_min))
        idx_max = np.argmin(np.abs(template[:, 0] - wav_max))

        templ_model = Model(template_model_new,
                            templ_disp=template[idx_min:idx_max, 0],
                            templ_flux=template[idx_min:idx_max, 1],
                            prefix=prefix)

    else:
        templ_model = Model(template_model_new,
                            templ_disp=template[:, 0],
                            templ_flux=template[:, 1],
                            prefix=prefix)



    templ_params.add(prefix +'z', value=redshift, min=0, max=1089)
    templ_params.add(prefix+'fwhm', value=fwhm)
    # Set amplitude to 1 initially and then calculate first  best guess flux
    # below
    templ_params.add(prefix+'amp', value=1.0e-4, min=1.0e-10, max=1.0)
    templ_params.add(prefix + 'intr_fwhm', value=intr_fwhm, vary=False)

    # normalize the template if possible
    if flux_2500 is not None:
        print("FLUX2500", flux_2500)
        if norm_wavelength is None:
            templ_params[prefix + 'amp'].set(value=flux_2500)
        else:
            model_flux = templ_model.eval(templ_params,
                                          x=norm_wavelength*(1.+redshift))
            print("Model_flux", model_flux)
            new_amp = flux_2500 / model_flux*0.5
            templ_params[prefix+'amp'].set(value=new_amp)

    else:
        templ_params[prefix + 'amp'].set(value=1.0e-15)

    return templ_model, templ_params

# ------------------------------------------------------------------------------
# Specialized Quasar Models
# ------------------------------------------------------------------------------

def CIII_model_func(x, z, cen, cen_alIII, cen_siIII,
                          amp, fwhm_km_s, shift_km_s,
                          amp_alIII, fwhm_km_s_alIII, shift_km_s_alIII,
                          amp_siIII, fwhm_km_s_siIII, shift_km_s_siIII):
    """

    :param x: ndarray
         x-Axis of the Gaussian
    :param z:  float
        Redshift for CIII], AlIII, SiIII]
    :param amp_cIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_cIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_cIII: float
        x-Axis shift of the Gaussian in km/s
    :param amp_alIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_alIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_alIII: float
        x-Axis shift of the Gaussian in km/s
    :param amp_siIII: float
        Amplitude of the Gaussian
    :param fwhm_km_s_siIII: float
        Full Width at Half Maximum (FWHM) of the Gaussian in km/s
    :param shift_km_s_siIII: float
            x-Axis shift of the Gaussian in km/s
    :return: float
    """

    cIII_cen = cen * (1 + z)
    siIII_cen = cen_siIII * (1 + z)
    alIII_cen = cen_alIII * (1 + z)

    cIII_delta_cen = shift_km_s / c_km_s * cIII_cen
    siIII_delta_cen = shift_km_s_siIII / c_km_s * siIII_cen
    alIII_delta_cen = shift_km_s_alIII / c_km_s * alIII_cen


    central = cIII_cen + cIII_delta_cen
    fwhm = fwhm_km_s / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_cIII = (amp / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = siIII_cen + siIII_delta_cen
    fwhm = fwhm_km_s_siIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_siIII = (amp_siIII / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    central = alIII_cen + alIII_delta_cen
    fwhm = fwhm_km_s_alIII / c_km_s * central
    sigma = fwhm / np.sqrt(8 * np.log(2))
    gauss_alIII = (amp_alIII / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -(x - central) ** 2 / (2 * sigma ** 2))

    return gauss_cIII + gauss_alIII + gauss_siIII


def subdivided_iron_template(fwhm=None, redshift=None, flux_2500=None,
                             templ_list=[]):

    # TODO Add Fe III templates

    param_list = []
    model_list = []
    if 'UV01' in templ_list or not templ_list:
        # 1200-1560 2
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV01_', flux_2500=flux_2500, wav_limits=[1200, 1560],
            norm_wavelength=1450)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV02' in templ_list or not templ_list:
        # 1560-1875 Vestergaard 2001
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV02_', flux_2500=flux_2500, wav_limits=[1560, 1875],
            norm_wavelength=1720)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV03' in templ_list or not templ_list:
        # 1875-2200 Vestergaard 2001
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV03_', flux_2500=flux_2500, wav_limits=[1875, 2200],
            norm_wavelength=2100)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV03_FeIII' in templ_list or not templ_list:
        templ_model, templ_params = load_template_model(
            template_filename='Fe3UV34modelB2.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV03_FeIII_', flux_2500=flux_2500, wav_limits=[1875,
                                                                   2200],
            norm_wavelength=1925)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV04' in templ_list or not templ_list:
        # 2200-2660 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV04_', flux_2500=flux_2500, wav_limits=[2200, 2660],
            norm_wavelength=2500)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV05' in templ_list or not templ_list:
        # 2660-3000 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV05_', flux_2500=flux_2500, wav_limits=[2660, 3000],
            norm_wavelength=2850)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV06' in templ_list or not templ_list:
        # 3000-3500 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
            prefix='UV06_', flux_2500=flux_2500, wav_limits=[3000, 3500],
            norm_wavelength=3250)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'OPT01' in templ_list or not templ_list:
        # 4400-4700 Boronson & Green 1992
        templ_model, templ_params = load_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT01_', flux_2500=flux_2500, wav_limits=[3700, 4700],
            norm_wavelength=4600)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'OPT02' in templ_list or not templ_list:
        # 4700-5100 Boronson & Green 1992
        templ_model, templ_params = load_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT02_', flux_2500=flux_2500, wav_limits=[4700, 5100],
            norm_wavelength=5000)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'OPT03' in templ_list or not templ_list:
        # 5100-5600 Boronson & Green 1992
        templ_model, templ_params = load_template_model(
            template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm,
            redshift=redshift,
            prefix='OPT03_', flux_2500=flux_2500, wav_limits=[5100, 5600],
            norm_wavelength=5200)
        param_list.append(templ_params)
        model_list.append(templ_model)
    # Vestergaard templates
    if 'UV04_V01' in templ_list or not templ_list:
        # 2200-2660 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV04V01_', flux_2500=flux_2500, wav_limits=[2200, 2660],
            norm_wavelength=2500)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV05_V01' in templ_list or not templ_list:
        # 2660-3000 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV05V01_', flux_2500=flux_2500, wav_limits=[2660, 3000],
            norm_wavelength=2850)
        param_list.append(templ_params)
        model_list.append(templ_model)
    if 'UV06_V01' in templ_list or not templ_list:
        # 3000-3500 Tsuzuki 2006
        templ_model, templ_params = load_template_model(
            template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
            prefix='UV06V01_', flux_2500=flux_2500, wav_limits=[3000, 3500],
            norm_wavelength=3250)
        param_list.append(templ_params)
        model_list.append(templ_model)


    return model_list, param_list


def iron_template_MgII(fwhm=None, redshift=None, flux_2500=None):

    # 2200-3500 Tsuzuki 2006
    templ_model, templ_params = load_template_model(
        template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
        prefix='FeIIMgII_', flux_2500=flux_2500, wav_limits=[2200, 3500],
        norm_wavelength=2500)

    return templ_model, templ_params

def iron_template_MgII_V01(fwhm=None, redshift=None, flux_2500=None):

    # 2200-3500 Vestergaard 2001
    templ_model, templ_params = load_template_model(
        template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
        prefix='FeIIMgII_', flux_2500=flux_2500, wav_limits=[2200, 3500],
        norm_wavelength=2500)

    return templ_model, templ_params

def iron_template_MgII_new(fwhm=None, redshift=None, flux_2500=None):

    # 2200-3500 Tsuzuki 2006
    templ_model, templ_params = load_template_model_new(
        template_filename='Tsuzuki06.txt', fwhm=fwhm, redshift=redshift,
        prefix='FeIIMgII_', flux_2500=flux_2500, wav_limits=[2200, 3500],
        norm_wavelength=2500, intr_fwhm=900)

    return templ_model, templ_params

def iron_template_MgII_V01_new(fwhm=None, redshift=None, flux_2500=None):

    # 2200-3500 Vestergaard 2001
    templ_model, templ_params = load_template_model_new(
        template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
        prefix='FeIIMgII_', flux_2500=flux_2500, wav_limits=[2200, 3500],
        norm_wavelength=2500, intr_fwhm=900)

    return templ_model, templ_params

def iron_template_CIV(fwhm=None, redshift=None, flux_2500=None):

    # 1200-2200 Vestergaard 2001
    templ_model, templ_params = load_template_model(
        template_filename='Fe_UVtemplt_A.asc', fwhm=fwhm, redshift=redshift,
        prefix='FeIICIV_', flux_2500=flux_2500, wav_limits=[1200, 2200],
        norm_wavelength=1450)

    return templ_model, templ_params

def iron_template_Hb(fwhm=None, redshift=None, flux_2500=None):

    # 3700-5600 Boronson & Green 1992
    templ_model, templ_params = load_template_model(
        template_filename='Fe_OPT_BR92_linear.txt', fwhm=fwhm, redshift=redshift,
        prefix='FeIIHb_', flux_2500=flux_2500, wav_limits=[3700, 5600],
        norm_wavelength=1450)

    return templ_model, templ_params


def balmer_continuum_model(x, z, flux_BE, T_e, tau_BE, lambda_BE):

    # Dietrich 2003
    # lambda <= 3646A, flux_BE = normalized estimate for Balmer continuum
    # The strength of the Balmer continuum can be estimated from the flux
    # density at 3675A after subtraction of the power-law continuum component
    # for reference see Grandi82, Wills85 or Verner99
    # at >= 3646A higher order balmer lines are merging  -> templates Dietrich03

    # The Balmer continuum does currently not include any blended high order
    # Balmer lines this has to be DONE!!!

    x = x / (1.+z)

    flux = flux_BE * blackbody_lambda(x, T_e).value * (
                1. - np.exp(-tau_BE * (x / lambda_BE) ** 3))

    flux[x >= lambda_BE] = flux[x>= lambda_BE] * 0

    flux *= 1e-20

    return flux


def create_line_model_HbOIII_6G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['hbeta_b_', 'hbeta_n_', 'OIII_a1_', 'OIII_a2_', 'OIII_b1_',
                'OIII_b2_']
    if flux_2500 is not None:
        amplitudes = np.array([20, 2, 5, 5, 5, 5]) * flux_2500
    else:
        amplitudes = np.array([20, 2, 5, 5, 5, 5]) * 1.0E-16

    widths = [2500, 900, 900, 900, 1200, 1200]
    central_wavs = [4862.68, 4862.68, 4960.30, 4960.30, 5008.24, 5008.24] #
    # VandenBerk2001
    shifts = [0, 0, 0, 0, 0, 0]

    param_list = []
    model_list = []


    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)


        param_list.append(params)
        model_list.append(model)


    for idx,params in enumerate(param_list):

        params[prefixes[idx] + 'amp'].set(min=redsh * 0.98, max=max(redsh *
                                                                    1.02, 1),
                     vary=True)
        params[prefixes[idx]+'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefixes[idx]+'shift_km_s'].set(vary=False, min=-200, max=200)
        params[prefixes[idx]+'cen'].set(expr=str(central_wavs[idx]))

    param_list[0]['hbeta_b_' + 'fwhm_km_s'].set(min=1200, max=20000)
    param_list[1]['hbeta_n_' + 'fwhm_km_s'].set(min=100, max=8000)
    param_list[2]['OIII_a1_' + 'fwhm_km_s'].set(min=100, max=2500)
    param_list[4]['OIII_b1_' + 'fwhm_km_s'].set(min=100, max=2500)
    param_list[3]['OIII_a2_' + 'fwhm_km_s'].set(min=500, max=6500)
    param_list[5]['OIII_b2_' + 'fwhm_km_s'].set(min=500, max=6500)

    # param_list[3]['OIII_a2_' + 'shift_km_s'].set(vary=True, min=-600, max=200)
    # param_list[5]['OIII_b2_' + 'shift_km_s'].set(vary=True, min=-600, max=200)

    return param_list, model_list


def create_line_model_HbOIII_4G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['hbeta_b_', 'hbeta_n_', 'OIII_a_', 'OIII_b_']
    if flux_2500 is not None:
        amplitudes = np.array([20, 2, 5, 5]) * flux_2500
    else:
        amplitudes = np.array([20, 2, 5, 5]) * 1.0E-16

    widths = [2500, 900, 1200, 1200]
    central_wavs = [4862.68, 4862.68, 4960.30, 5008.24]
    shifts = [0, 0, 0, 0]

    param_list = []
    model_list = []


    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)


        param_list.append(params)
        model_list.append(model)


    for idx,params in enumerate(param_list):

        params[prefixes[idx] + 'amp'].set(min=redsh * 0.98, max=max(redsh *
                                                                    1.02, 1),
                     vary=True)
        params[prefixes[idx]+'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefixes[idx]+'shift_km_s'].set(vary=False, min=-200, max=200)
        params[prefixes[idx]+'cen'].set(expr=str(central_wavs[idx]))

    param_list[0]['hbeta_b_' + 'fwhm_km_s'].set(min=1200, max=20000)
    param_list[1]['hbeta_n_' + 'fwhm_km_s'].set(min=100, max=8000)
    param_list[2]['OIII_a_' + 'fwhm_km_s'].set(min=100, max=6000)
    param_list[3]['OIII_b_' + 'fwhm_km_s'].set(min=500, max=6000)


    return param_list, model_list


def create_line_model_Hb_2G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['hbeta_b_', 'hbeta_n_']
    if flux_2500 is not None:
        amplitudes = np.array([20, 2]) * flux_2500
    else:
        amplitudes = np.array([20, 2]) * 1.0E-16

    widths = [2500, 900]
    central_wavs = [4862.68, 4862.68]
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)


    for idx,params in enumerate(param_list):

        params[prefixes[idx]+'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefixes[idx]+'shift_km_s'].set(vary=False, min=-200, max=200)
        params[prefixes[idx]+'cen'].set(expr=str(central_wavs[idx]))

    param_list[0]['hbeta_b_' + 'fwhm_km_s'].set(min=1200, max=20000)
    param_list[1]['hbeta_n_' + 'fwhm_km_s'].set(min=100, max=8000)

    return param_list, model_list



def create_line_model_Ha_2G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['halpha_b_', 'halpha_n_']
    if flux_2500 is not None:
        amplitudes = np.array([20, 2]) * flux_2500
    else:
        amplitudes = np.array([20, 2]) * 1.0E-16

    widths = [10000, 5000]
    central_wavs = [6564.61, 6564.61]
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)


    for idx,params in enumerate(param_list):

        params[prefixes[idx]+'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefixes[idx]+'shift_km_s'].set(vary=False, min=-200, max=200)
        params[prefixes[idx]+'cen'].set(expr=str(central_wavs[idx]))

    param_list[0]['halpha_b_' + 'fwhm_km_s'].set(min=1200, max=30000)
    param_list[1]['halpha_n_' + 'fwhm_km_s'].set(min=100, max=10000)

    return param_list, model_list


def create_line_model_Ha_3G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['halpha_a_', 'halpha_b_', 'halpha_n_']
    if flux_2500 is not None:
        amplitudes = np.array([10, 20, 2]) * flux_2500
    else:
        amplitudes = np.array([10, 20, 2]) * 1.0E-16

    widths = [10000, 5000, 500]
    central_wavs = [6564.61, 6564.61, 6564.61]
    shifts = [0, 0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)


    for idx,params in enumerate(param_list):

        params[prefixes[idx]+'amp'].set(min=1.0e-19, max=1.0e-10)
        params[prefixes[idx]+'shift_km_s'].set(vary=False, min=-200, max=200)
        params[prefixes[idx]+'cen'].set(expr=str(central_wavs[idx]))

    param_list[0]['halpha_a_' + 'fwhm_km_s'].set(min=1200, max=30000)
    param_list[1]['halpha_b_' + 'fwhm_km_s'].set(min=100, max=10000)
    param_list[2]['halpha_n_' + 'fwhm_km_s'].set(min=50, max=2000)

    return param_list, model_list



def create_line_model_MgII_2G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['mgII_b_', 'mgII_n_']

    if flux_2500 is not None:
        amplitudes = np.array([20, 2]) * flux_2500
    else:
        amplitudes = np.array([20, 2]) * 1.0E-16

    widths = [2500, 1000]
    central_wavs = [2798.75, 2798.75] # Vanden Berk 2001
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)

    param_list[0]['mgII_b_' + 'cen'].set(vary=False)
    param_list[1]['mgII_n_' + 'cen'].set(vary=False)

    param_list[0]['mgII_b_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['mgII_b_' + 'shift_km_s'].set(vary=False, min=-200, max=200)
    param_list[0]['mgII_b_' + 'fwhm_km_s'].set(min=1200, max=10000)

    param_list[1]['mgII_n_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[1]['mgII_n_' + 'shift_km_s'].set(vary=False, min=-200, max=200)
    param_list[1]['mgII_n_' + 'fwhm_km_s'].set(min=100, max=1200)

    return param_list, model_list


def create_line_model_MgII_1G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['mgII_']

    if flux_2500 is not None:
        amplitudes = np.array([20, 2]) * flux_2500
    else:
        amplitudes = np.array([20, 2]) * 1.0E-16

    widths = [2500]
    central_wavs = [2798.75] # Vanden Berk 2001
    shifts = [0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):

        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)


    param_list[0]['mgII_' + 'cen'].set(vary=False)
    # param_list[0]['mgII_' + 'z'].set(expr='z')
    param_list[0]['mgII_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['mgII_' + 'shift_km_s'].set(vary=False, min=-200, max=200)
    param_list[0]['mgII_' + 'fwhm_km_s'].set(min=1200, max=10000)


    return param_list, model_list


def create_line_model_CIV_2G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['cIV_b_', 'cIV_n_']

    if flux_2500 is not None:
        amplitudes = np.array([20, 10]) * flux_2500
    else:
        amplitudes = np.array([20, 10]) * 1.0E-16

    widths = [2500, 2500]
    central_wavs = [1549.06, 1549.06] # Vanden Berk 2001
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):
        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)


    print(params)

    param_list[0]['cIV_b_' + 'cen'].set(vary=False)
    param_list[0]['cIV_b_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['cIV_b_' + 'shift_z'].set(vary=False, min=-200, max=200)
    param_list[0]['cIV_b_' + 'fwhm_km_s'].set(min=1200, max=20000)

    param_list[1]['cIV_n_' + 'cen'].set(vary=False)
    param_list[1]['cIV_n_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[1]['cIV_n_' + 'shift_z'].set(vary=False, min=-200, max=200)
    param_list[1]['cIV_n_' + 'fwhm_km_s'].set(min=1200, max=20000)

    return param_list, model_list


def create_line_model_CIV_1G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['cIV_']

    if flux_2500 is not None:
        amplitudes = np.array([20]) * flux_2500
    else:
        amplitudes = np.array([20]) * 1.0E-16

    widths = [2500]
    central_wavs = [1549.06] # Vanden Berk 2001
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):
        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)

    param_list[0]['cIV_' + 'cen'].set(vary=False)
    param_list[0]['cIV_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['cIV_' + 'shift_z'].set(vary=False, min=-200, max=200)
    param_list[0]['cIV_' + 'fwhm_km_s'].set(min=1200, max=30000)


    return param_list, model_list


def create_line_model_CIII_1G(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['cIII_']

    if flux_2500 is not None:
        amplitudes = np.array([20]) * flux_2500
    else:
        amplitudes = np.array([20]) * 1.0E-16

    widths = [2500]
    central_wavs = [1908.73] # Vanden Berk 2001
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):
        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.9, max=max(redsh * 1.1, 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.9, max=max(redsh * 1.1, 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)

    param_list[0]['cIII_' + 'cen'].set(vary=False)
    param_list[0]['cIII_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['cIII_' + 'shift_z'].set(vary=False, min=-200, max=200)
    param_list[0]['cIII_' + 'fwhm_km_s'].set(min=1200, max=30000)


    return param_list, model_list


def create_line_model_CIII(fit_z=True, redsh=0.0, flux_2500=None):


    if flux_2500 is not None:
        amp_cIII = 21.19 * flux_2500
        amp_alIII = 0.4 * flux_2500
        amp_siIII = 0.16 * flux_2500
    else:
        amp_cIII = 21.19 * 1.0E-16
        amp_alIII = 0.4 * 1.0E-16
        amp_siIII = 0.16 * 1.0E-16

    params = Parameters()

    params.add('cIII_cen', value=1908.73, vary=False)
    params.add('cIII_cen_siIII', value=1892.03, vary=False)
    params.add('cIII_cen_alIII', value=1857.40, vary=False)

    if fit_z:
        params.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02, 1),
                 vary=True)
    else:
        params.add('z', value=redsh, min=redsh * 0.98, max=max(redsh * 1.02, 1),
                 vary=False)

    params.add('cIII_z', value=redsh, vary=True)

    params.add('cIII_amp', value=amp_cIII, vary=True, min=1e-19, max=1e-10)
    params.add('cIII_amp_alIII', value=amp_alIII, vary=True, min=1e-19,
               max=1e-10)
    params.add('cIII_amp_siIII', value=amp_siIII, vary=True, min=1e-19,
               max=1e-10)

    params.add('cIII_fwhm_km_s', value=2000, vary=True, min=500, max=1e+4)
    params.add('cIII_fwhm_km_s_alIII', value=400, vary=True, min=500, max=7e+3)
    params.add('cIII_fwhm_km_s_siIII', value=300, vary=True, min=500, max=7e+3)

    params.add('cIII_shift_km_s', value=0, vary=False)
    params.add('cIII_shift_km_s_alIII', value=0, vary=False)
    params.add('cIII_shift_km_s_siIII', value=0, vary=False)

    elmodel = Model(CIII_model_func, prefix='cIII_')

    return params, elmodel


def create_line_model_HeII_HighZ(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['heII_']

    if flux_2500 is not None:
        amplitudes = np.array([20]) * flux_2500
    else:
        amplitudes = np.array([20]) * 1.0E-16

    widths = [2500]
    central_wavs = [1640.42] # Vanden Berk 2001
    shifts = [0, 0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):
        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)

    param_list[0]['heII_' + 'cen'].set(vary=False)
    param_list[0]['heII_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['heII_' + 'shift_km_s'].set(vary=False, min=-200, max=200)
    param_list[0]['heII_' + 'fwhm_km_s'].set(min=1200, max=10000)

    return param_list, model_list



def create_line_model_SiIV_HighZ(fit_z=False, redsh=0.0, flux_2500=None):

    prefixes = ['siIV_']

    if flux_2500 is not None:
        amplitudes = np.array([20]) * flux_2500
    else:
        amplitudes = np.array([20]) * 1.0E-16

    widths = [2500]
    central_wavs = [1399.8 ] # SiIV + OIV] http://classic.sdss.org/dr6/algorithms/linestable.html
    shifts = [0]

    param_list = []
    model_list = []

    for idx, prefix in enumerate(prefixes):
        pars = Parameters()

        if fit_z:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=True)
        else:
            pars.add('z', value=redsh, min=redsh * 0.95, max=max(redsh * 1.05,
                                                                 1),
                     vary=False)

        params, model = emission_line_model(amp=amplitudes[idx],
                                            cen=central_wavs[idx],
                                            wid=widths[idx],
                                            shift=shifts[idx],
                                            unit_type='fwhm_km_s_z',
                                            prefix=prefix,
                                            fit_central=True,
                                            parameters=pars,
                                            redsh=redsh)

        param_list.append(params)
        model_list.append(model)

    param_list[0]['siIV_' + 'cen'].set(vary=False)
    param_list[0]['siIV_' + 'amp'].set(min=1.0e-19, max=1.0e-10)
    param_list[0]['siIV_' + 'shift_km_s'].set(vary=False, min=-200, max=200)
    param_list[0]['siIV_' + 'fwhm_km_s'].set(min=1200, max=10000)

    return param_list, model_list
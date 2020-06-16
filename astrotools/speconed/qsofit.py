
import os
import glob
import numpy as np
import lmfit
import corner
import pandas as pd

from lmfit import Model, Parameters
from lmfit.model import save_model, load_model, save_modelresult, load_modelresult
from lmfit.models import ExponentialModel, GaussianModel, LinearModel, VoigtModel

import matplotlib.pyplot as plt
from matplotlib import rc

from astrotools.speconed import speconed as sod
from astrotools import cosmology as cosm
from astrotools import photometric_functions as phot

import astropy.constants as const
import astropy.units as units

# from astrotools.speconed import specfit_models as specmod

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel, QHBoxLayout, QLineEdit, QCheckBox, QFileDialog, QComboBox,  QScrollArea, QGroupBox

import matplotlib.transforms as mtransforms

pc_in_cm = units.pc.to(units.cm)
c_km_s = const.c.to('km/s').value
L_sun_erg_s = const.L_sun.to('erg/s').value

from scipy.interpolate import UnivariateSpline

import astrotools.speconed.specfit_models as specmod

import linetools.utils as lu

black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (204/255., 121/255., 167/255.)



# ------------------------------------------------------------------------------
# MODEL FUNCTIONS
# ------------------------------------------------------------------------------

def load_qso_fit(foldername):

    # Quick and dirty initialization of in_dict
    in_dict = {'spec': None,
                'mask_list': [None, None, None],
                'cont_fit_spec': None,
                'cont_init_spec': None,
                'line_fit_spec': None,
                'line_init_spec': None,
                'x_lo': None,
                'x_hi': None,
                'y_lo': None,
                'y_hi': None,
                'line_model_list': [],
                'line_model_par_list': [],
                'cont_model_list': [],
                'cont_model_par_list': [],
                'cont_fit_result': None,
                'line_fit_result': None}

    # Read the hdf5 data extension and populate spectra, masks and fits
    df = pd.read_hdf(foldername + '/fit.hdf5', 'data')

    disp = df['dispersion'].values
    flux = df['flux'].values

    in_dict['spec'] = sod.SpecOneD(dispersion=disp, flux=flux, unit='f_lam')

    if hasattr(df, 'flux_err'):
        flux_err = df['flux_err'].values
        in_dict['spec'].flux_err = flux_err

    spec = in_dict['spec']

    in_dict['mask_list'][0] = np.array(df['mask_spec'].values,
                                            dtype=bool)
    in_dict['mask_list'][1] = np.array(df['mask_cont'].values,
                                            dtype=bool)
    in_dict['mask_list'][2] = np.array(df['mask_emline'].values,
                                            dtype=bool)

    spec.mask = in_dict['mask_list'][0]

    if hasattr(df, 'cont_fit_spec'):
        cont_fit_flux = df['cont_fit_spec'].values
        cont_fit_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=cont_fit_flux, unit='f_lam')
        in_dict['cont_fit_spec'] = cont_fit_spec

    if hasattr(df, 'cont_init_spec'):
        cont_init_flux = df['cont_init_spec'].values
        cont_init_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=cont_init_flux, unit='f_lam')
        in_dict['cont_init_spec'] = cont_init_spec

    if hasattr(df, 'line_fit_spec'):
        line_fit_flux = df['line_fit_spec'].values
        line_fit_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=line_fit_flux, unit='f_lam')
        in_dict['line_fit_spec'] = line_fit_spec

    if hasattr(df, 'line_init_spec'):
        line_init_flux = df['line_init_spec'].values
        line_init_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=line_init_flux, unit='f_lam')
        in_dict['line_init_spec'] = line_init_spec

    # # Read the hdf5 parameter extension and populate specfitgui parameters
    df = pd.read_hdf(foldername + '/fit.hdf5', 'params')
    cont_fit_z_flag = df.loc['cont_fit_z_flag', 'value']
    line_fit_z_flag = df.loc['line_fit_z_flag', 'value']

    in_dict['cont_fit_z_flag'] = cont_fit_z_flag
    in_dict['line_fit_z_flag'] = line_fit_z_flag
    in_dict['fit_with_weights'] = df.loc['fit_with_weights', 'value']


    # Create the model list and read the json files for the continuum and line models
    cont_model_list = []
    cont_model_par_list = []

    line_model_list = []
    line_model_par_list = []

    model_dict = {'power_law': specmod.power_law,
                  'power_law_continuum': specmod.power_law,
                  'power_law_at_2500A': specmod.power_law_at_2500A,
                  'template_model': specmod.template_model,
                  'gaussian_fwhm_km_s': specmod.gaussian_fwhm_km_s,
                  'gaussian_fwhm_km_s_z': specmod.gaussian_fwhm_km_s_z,
                  'gaussian_fwhm_z': specmod.gaussian_fwhm_z,
                  'balmer_continuum_model': specmod.balmer_continuum_model,
                  'power_law_at_2500A_plus_BC':
                      specmod.power_law_at_2500A_plus_BC,
                  'power_law_at_2500A_plus_flexible_BC':
                      specmod.power_law_at_2500A_plus_flexible_BC,
                  'CIII_model_func': specmod.CIII_model_func}

    cont_models = glob.glob(foldername + '/cont_*.json')

    for model_fname in cont_models:
        model = load_model(model_fname, funcdefs=model_dict)

        # if cont_fit_z_flag:
        params = Parameters()
        params.add('z', value=0, min=0, max=1000, vary=True)
        pars = model.make_params()
        for p in pars:
            params.add(pars[p])
        # else:
        #     params = model.make_params()

        cont_model_list.append(model)
        cont_model_par_list.append(params)

    line_models = glob.glob(foldername + '/line_*.json')

    for model_fname in line_models:
        model = load_model(model_fname, funcdefs=model_dict)

        # if line_fit_z_flag:
        params = Parameters()
        params.add('z', value=0, min=0, max=1000, vary=True)
        pars = model.make_params()
        for p in pars:
            params.add(pars[p])
        # else:
        #     params = model.make_params()

        line_model_list.append(model)
        line_model_par_list.append(params)

    in_dict['cont_model_list'] = cont_model_list
    in_dict['line_model_list'] = line_model_list

    if os.path.isfile(foldername + '/fit_cont_result.json'):
        cont_fit_result = load_modelresult(foldername + '/fit_cont_result.json', funcdefs=model_dict)
        in_dict['cont_model_par_list'] = upd_param_values_from_fit(
            cont_model_list, cont_model_par_list, cont_fit_result)
        in_dict['cont_fit_result'] = cont_fit_result


    if os.path.isfile(foldername + '/fit_lines_result.json'):
        line_fit_result = load_modelresult(foldername + '/fit_lines_result.json', funcdefs=model_dict)
        in_dict['line_model_par_list'] = upd_param_values_from_fit(
            line_model_list, line_model_par_list, line_fit_result)
        in_dict['line_fit_result'] = line_fit_result

    return in_dict


def build_continuum_model(cont_model_list, cont_model_par_list,
                          cont_fit_z_flag=False, redsh_par=None, z_vary=None):


    """ Build the full continuum model from the models and parameters
    in the continnum model and parameter lists.

    This is the model that is fit.

    """

    cont_model_pars = Parameters()
    # For all model parameters in the parameter model list
    for idx, params in enumerate(cont_model_par_list):
        # For all parameters within the model parameters
        for jdx, p in enumerate(params):
            # Add the parameters to the continuum parameters
            cont_model_pars.add(p, expr=params[p].expr,
                                value=params[p].value,
                                min=params[p].min,
                                max=params[p].max,
                                vary=params[p].vary)

    # Add the redshift parameter if included in the fit
    if cont_fit_z_flag:
        if z_vary is not None:
            vary = z_vary
        elif redsh_par is not None:
            vary = redsh_par['z'].vary
        else:
            vary = False

        if 'z' in cont_model_pars:
            cont_model_pars['z'].set(vary=vary)
        else:
            cont_model_pars.add(redsh_par['z'],
                                value=redsh_par['z'].value,
                                min=redsh_par['z'].min,
                                max=redsh_par['z'].max,
                                vary=vary)

    # Build the continuum model from the individual models
    if len(cont_model_list) > 0:
        cont_model = cont_model_list[0]

        for cm in cont_model_list[1:]:
            cont_model += cm

        return cont_model, cont_model_pars


def build_line_model(line_model_list, line_model_par_list,
                     line_fit_z_flag=False,
                     redsh_par=None, z_vary=None):
    """ Build the full line model from the models and parameters
            in the line model and parameter lists.

            This is the model that is fit.

            """

    line_model_pars = Parameters()

    # For each set of model parameters in the model parameter set list
    for idx, params in enumerate(line_model_par_list):
        # For each parameter in the model parameter set
        for jdx, p in enumerate(params):
            print(p, params[p].vary)
            # Add parameter to full line model parameters
            line_model_pars.add(p, expr=params[p].expr,
                                value=params[p].value,
                                min=params[p].min,
                                max=params[p].max,
                                vary=params[p].vary)


    # Add the redshift to the fit, if selected
    if line_fit_z_flag:

        if z_vary is not None:
            vary = z_vary
        elif redsh_par is not None:
            vary = redsh_par['z'].vary
        else:
            vary = False

        if 'z' in line_model_pars:
            line_model_pars['z'].set(vary=vary)
        else:
            line_model_pars.add(redsh_par['z'],
                                value=redsh_par['z'].value,
                                min=redsh_par['z'].min,
                                max=redsh_par['z'].max,
                                vary=vary)

        line_model_pars['z'].set(value=redsh_par['z'].value, vary=vary)


    # Add all models in the model list to the full line model
    if len(line_model_list) > 0:
        line_model = line_model_list[0]

        for lm in line_model_list[1:]:
            line_model += lm

        return line_model, line_model_pars


def upd_param_values_from_fit(model_list, model_par_list, fit_result):

    # update the redshift parameter
    # print('fit params', fit_result.params)


    # update all parameters in model_par_list
    for idx, model in enumerate(model_list):

        params = model_par_list[idx]

        for jdx, param in enumerate(params):

            temp_val = fit_result.params[param].value
            model_par_list[idx][param].value = temp_val
            # print (temp_val)

            temp_val = fit_result.params[param].expr
            model_par_list[idx][param].expr = temp_val
            # print(temp_val)

            temp_val = fit_result.params[param].min
            model_par_list[idx][param].min = temp_val
            # print(temp_val)

            temp_val = fit_result.params[param].max
            model_par_list[idx][param].max = temp_val
            # print(temp_val)

            temp_val = fit_result.params[param].vary
            model_par_list[idx][param].vary = temp_val
            # print(temp_val)

    return model_par_list

# ------------------------------------------------------------------------------
# ANALYSIS FUNCTIONS
# ------------------------------------------------------------------------------

def build_line_flux_from_line_models(dispersion, line_prefix_list, line_model_list,
                               line_model_pars):


    line_flux = np.zeros(len(dispersion))

    for idx, line in enumerate(line_model_list):
        if line.prefix in line_prefix_list:
            try:
                params = line_model_pars[idx]
            except:
                params = line_model_pars

            line_flux += line.eval(params, x=dispersion)

    return line_flux


def build_cont_flux_from_cont_models(dispersion,
                                     cont_model_list, cont_model_pars):

    model, pars = build_continuum_model(cont_model_list, cont_model_pars)

    cont_flux = model.eval(pars, x=dispersion)

    return cont_flux


def calc_Edd_luminosity(bh_mass):

    factor = (4 * np.pi * const.G * const.c * const.m_p) / const.sigma_T
    factor = factor.to(units.erg / units.s / units.Msun).value

    return factor * bh_mass



def calc_peak_from_line(dispersion, flux):
    pass

def calc_fwhm_from_line(dispersion, flux):

    spline = UnivariateSpline(dispersion, flux - np.max(flux) / 2., s=0)

    roots = spline.roots()

    if len(roots) > 2 or len(roots) < 2:
        print('[WARNING] Found {} roots. Cannot determine FWHM'.format(len(
            roots)))
        print('[WARNING] FWHM set to NaN')
        # plt.cla()
        # plt.plot(dispersion, flux - np.max(flux) / 2., 'r-', lw=2)
        # plt.plot(dispersion, dispersion*0, 'k--')
        # plt.show()
        return [np.NaN]
    else:

        max_idx = np.where(flux == np.max(flux))
        max_wav = dispersion[max_idx]

        # plt.plot(dispersion, line_flux)
        # # spline = UnivariateSpline(dispersion, line_flux, s=0)
        # plt.plot(dispersion, spline(dispersion))
        # plt.show()

        fwhm = (abs(roots[0] - max_wav) + abs(roots[1] - max_wav)) / max_wav * \
               c_km_s

        # print("FWHM:", fwhm)

        return fwhm

def calc_integrated_flux(dispersion, flux, range=None):

    spec = sod.SpecOneD(dispersion=dispersion, flux=flux,
                        unit='f_lam')

    if range is not None:
        spec.trim_dispersion(range, inplace=True)

    return np.trapz(spec.flux, x=spec.dispersion)


def calc_integrated_line_luminosity_new(dispersion, flux,
                                    redshift, cosmology, range=None):
    # convert to rest-frame flux and dispersion
    rest_dispersion = dispersion / (1. + redshift)
    rest_flux = calc_Lwav_from_fwav(flux, redshift, cosmology)
    spec = sod.SpecOneD(dispersion=rest_dispersion, flux=rest_flux,
                        unit='f_lam')

    if range is not None:
        spec.trim_dispersion(range, inplace=True)

    integrated_flux = np.trapz(spec.flux, x=spec.dispersion)

    return integrated_flux

def calc_integrated_line_luminosity(dispersion, line_flux,
                                    redshift, cosmology):

    # convert to rest-frame flux and dispersion
    rest_dispersion = dispersion / (1. + redshift)
    rest_flux = calc_Lwav_from_fwav(line_flux, redshift, cosmology)

    integrated_flux = np.trapz(rest_flux, x=rest_dispersion)

    return integrated_flux


def calc_fwhm_from_line_models(dispersion, line_prefix_list, line_model_list,
                               line_model_pars):

    line_flux = build_line_flux_from_line_models(dispersion, line_prefix_list, line_model_list,
                               line_model_pars)

    fwhm = calc_fwhm_from_line(dispersion, line_flux)

    return fwhm



def calc_equivalent_width_from_line_models(dispersion, cont_flux, line_flux,
                                           limits=None, redshift=None):


    if redshift is not None:
        rest_dispersion = dispersion / (1+redshift)
        rest_cont_flux = cont_flux * (1+redshift)
        rest_line_flux = line_flux * (1+redshift)

    if limits is not None:
        l_idx = np.argmin(np.abs(rest_dispersion - limits[0]))
        u_idx = np.argmin(np.abs(rest_dispersion - limits[1]))

        ew = np.trapz((rest_line_flux[l_idx:u_idx])/rest_cont_flux[l_idx:u_idx],
                      rest_dispersion[l_idx:u_idx])
    else:
        ew = np.trapz((rest_line_flux) / rest_cont_flux,
                      rest_dispersion)

    return ew


def calc_Lwav_from_fwav(f_wav, redsh, cos):

    """Calculate the monochromatic luminosity from the monochromativ flux
    density.

    Parameters:
    -----------
    f_wav : float
        Monochromatic flux density in units of erg s^-1 cm^-2 Angstroem^-1

    redsh : float
        Redshift of the object in question

    cos : Cosmology()
        User defined cosmology given as an object of the class Cosmology()

    Returns:
    --------
    L_wav : float
        Monochromatic luminosity at the specified wavelength in units of
        erg s^-1 Angstroem^-1
    """

    # Calculate luminosity distance in Mpc
    lum_distance = cos.luminosity_distance(redsh)/cos.h
    # Convert to cm
    lum_distance *= pc_in_cm * 1e+6

    # TODO Check if slope really enters here!!!
    return f_wav * lum_distance**2 * 4 * np.pi * (1. + redsh)


def calc_abmag_from_fwav(f_wav, wav):

    f_nu = f_wav * (wav)**2 / 2.998e+18

    return -2.5*np.log10(f_nu) - 48.6


def calc_QSO_Lbol_from_L3000(L3000):

    return L3000*5.15



def calc_QSO_Lbol_Ne19(L_wav, wav):
    """
    Derive bolometric luminosities for quasars using the Netzer 2019 bolometric
    corrections. Values are taken from Table 1 in Netzer 2019.
    :param L_wav:
    :param wav:
    :return:
    """

    if wav in [1400, 3000, 5100]:

        if wav == 1400:
            c = 7
            d = -0.1
        elif wav == 3000:
            c = 19
            d = -0.2
        elif wav == 5100:
            c = 40
            d = -0.2

        k_bol = c * (L_wav/ 10 ** 42)**d

        return L_wav * k_bol

    else:
        print('[ERROR] Bolometric correction for specified wavelength not '
              'available.')


def correct_CIV_fwhm(fwhm, blueshift):
    """
    Correct the CIV FWHM according to Coatman et al. 2017 Eq. 4

    :param fwhm: float
        CIV FWHM in km/s
    :param blueshift: float
        CIV blueshift in km/s (not velocity shift, use correct sign)
    :return:
    """

    return fwhm / (0.41 * blueshift/1000 + 0.62)


def calc_CIV_BHmass_Co17(L_wav, wav, fwhm, verbosity=1):

    reference = "Co17"

    if wav == 1350:
        return 10 ** 6.71 * (fwhm / 1000.) ** 2 * (
                    wav * L_wav / 10 ** 44) ** (0.53), reference
    else:
        if verbosity > 1:
            print("Specified wavelength does not allow for BH mass "
                  "calculation with Hbeta", wav)


def calc_BHmass_VP06VO09(L_wav, wav, fwhm, line="MgII", verbosity=2):

    """ Calculate black hole mass according to the empirical relations
    of Vestergaard & Peterson 2006 and Vestergaard & Osmer 2009

    See equation (1) in Vestergaard & Osmer 2009 for MgII
    See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV

    Parameters:
    -----------
    L_wav : float
        Monochromatic luminosity at the given wavelength of wav in erg s^-1
        Angstroem^-1

    wav : float or int
        Wavelength for the given monochromatic luminosity

    fwhm : float
        Full width at half maximum of the emission line specified in line

    line : string
        Name of the emission line to calculate the black hole mass from. Possible
        line names are MgII, Hbeta, CIV

    Returns:
    --------


    BHmass : float
    Black hole mass in units for solar masses

    """
    zp = None

    if line == "MgII":
        reference = "VW09"

        if wav == 1350:
            zp = 6.72

        elif wav == 2100:
            zp = 6.79

        elif wav == 3000:
            zp = 6.86

        elif wav == 5100:
            zp = 6.96

        else:
            if verbosity > 1:
                print("Specified wavelength does not allow for BH mass "
                       "calculation with MgII", wav)
        if zp is not None:
            return 10**zp * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
               reference
        # else:
        #     return None, None

    elif line == "Hbeta":
        reference = "VO06"

        if wav == 5100:
            return 10**6.91 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
                   reference

        else:
            if verbosity > 1:
                print ("Specified wavelength does not allow for BH mass "
                       "calculation with Hbeta", wav)
            # return None, None

    elif line == "CIV":
        reference = "VO06"

        if wav == 1350:
            return 10**6.66 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.53),\
                   reference

        else:
            if verbosity > 1:
                print ("Specified wavelength does not allow for BH mass "
                       "calculation with CIV", wav)

    else:
        if verbosity >1:
            print("[Warning] No relation exists to calculate the BH mass for "
                  "the specified line ({}) and wavelength ({}): ".format(
                line, wav))

    return None, None

def calc_BH_masses(L_wav, wav, fwhm, line="MgII", verbosity=2):

    """ Calculate black hole mass according to the empirical relations
    of Vestergaard & Peterson 2006 and Vestergaard & Osmer 2009

    See equation (1) in Vestergaard & Osmer 2009 for MgII
    See equations (5, 7) in Vestergaard & Peterson for Hbeta and CIV

    Parameters:
    -----------
    L_wav : float
        Monochromatic luminosity at the given wavelength of wav in erg s^-1
        Angstroem^-1

    wav : float or int
        Wavelength for the given monochromatic luminosity

    fwhm : float
        Full width at half maximum of the emission line specified in line

    line : string
        Name of the emission line to calculate the black hole mass from. Possible
        line names are MgII, Hbeta, CIV

    Returns:
    --------


    BHmass : float
    Black hole mass in units for solar masses

    """
    zp = None
    b = None

    if line == "MgII":
        reference = "VW09"

        if wav == 1350:
            zp = 6.72

        elif wav == 2100:
            zp = 6.79

        elif wav == 3000:
            zp = [6.86, 6.74]
            b = [0.5, 0.62]
            reference = ['VW09', 'S11']

        elif wav == 5100:
            zp = 6.96

        else:
            if verbosity > 1:
                print("Specified wavelength does not allow for BH mass "
                       "calculation with MgII", wav)
        if zp is not None and b is None:
            return 10**zp * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
               reference
        elif zp is not None and b is not None:

            bhmass_a = 10**zp[0] * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(
                b[0])
            bhmass_b = 10 ** zp[1] * (fwhm / 1000.) ** 2 * (
                        wav * L_wav / 10 ** 44) ** (
                           b[1])
            bhmass = [bhmass_a, bhmass_b]

            return bhmass, reference



    elif line == "Hbeta":
        reference = "VO06"

        if wav == 5100:
            return 10**6.91 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.5), \
                   reference

        else:
            if verbosity > 1:
                print ("Specified wavelength does not allow for BH mass "
                       "calculation with Hbeta", wav)
            # return None, None

    elif line == "CIV":
        reference = "VO06"

        if wav == 1350:
            if not np.isnan(fwhm):
                return 10**6.66 * (fwhm/1000.)**2 * (wav * L_wav / 10**44)**(0.53),\
                       reference
            else:
                return np.NaN, reference

        else:
            if verbosity > 1:
                print ("Specified wavelength does not allow for BH mass "
                       "calculation with CIV", wav)

    else:
        if verbosity > 1:
            print("[Warning] No relation exists to calculate the BH mass for "
                  "the specified line ({}) and wavelength ({}): ".format(
                line, wav))



    return None, None


def calc_Halpha_BH_mass(L_Halpha, FWHM_Halpha):

    # Implementation after Greene & Ho 2005 Equation 6
    reference = "GH05"

    if not np.isnan(FWHM_Halpha) or not np.isnan(L_Halpha):

        return 2.0 * 1e+6 *  (FWHM_Halpha/1000)**2.06 * (L_Halpha/
                                                       10**42)**0.55, \
               reference



# ------------------------------------------------------------------------------
# MAIN RE-FITTING AND ANALYSIS ROUTINES
# ------------------------------------------------------------------------------


def fit_emcee(filename, foldername, redshift):
    in_dict = load_qso_fit(foldername)

    if hasattr(in_dict['spec'], 'flux_err') == False:
        raise ValueError("The spectrum does not have usable flux errors")

    # read original spectrum
    spec = sod.SpecOneD()
    spec.read_pypeit_fits(filename)

    cont_model_list = in_dict['cont_model_list']
    line_model_list = in_dict['line_model_list']
    cont_model_par_list = in_dict['cont_model_par_list']
    line_model_par_list = in_dict['line_model_par_list']

    cont_fit_z_flag = in_dict['cont_fit_z_flag']
    line_fit_z_flag = in_dict['line_fit_z_flag']

    # TODO getting z from line_model_par_list will cause problems,
    #  have general
    #  redshift parameter instead
    redsh_par = Parameters()
    redsh_par.add(line_model_par_list[0]['z'])
    redsh_par.value = redshift

    redsh_par_cont = Parameters()
    redsh_par_cont.add(line_model_par_list[0]['z'])
    redsh_par_cont.value = redshift

    # 0) Build line and continuum model

    cont_model, cont_model_pars = build_continuum_model(cont_model_list,
                                                        cont_model_par_list,
                                                        cont_fit_z_flag,
                                                        redsh_par_cont,
                                                        z_vary=False)

    line_model, line_model_pars = build_line_model(line_model_list,
                                                   line_model_par_list,
                                                   line_fit_z_flag,
                                                   redsh_par,
                                                   z_vary=True)


    # 1) Fit using the pre-defined parameters
    spec.smooth(40, inplace=True)
    cont_m = np.logical_and(in_dict["mask_list"][1], in_dict[
        "mask_list"][0])
    line_m = np.logical_and(in_dict["mask_list"][2], in_dict[
        "mask_list"][0])
    cont_fit_result = cont_model.fit(spec.flux[cont_m],
                                     cont_model_pars,
                                     x=spec.dispersion[cont_m],
                                     weights=1./spec.flux_err[cont_m]**2)

    cont_fit_flux = cont_model.eval(cont_fit_result.params,
                                    x=spec.dispersion)

    res_flux = spec.flux - cont_fit_flux

    line_fit_result = line_model.fit(res_flux[line_m],
                                     line_model_pars,
                                     x=spec.dispersion[line_m],
                                     weights=1./spec.flux_err[line_m]**2)

    line_fit_flux = line_model.eval(line_fit_result.params,
                                    x=spec.dispersion)

    # 2) Use guesses to set up the emcee sampler
    # print(cont_model_pars)
    # plt.plot(spec.dispersion[cont_m], spec.flux[cont_m])
    # plt.show()

    emcee_kws = dict(steps=10000, burn=1000)
    emcee_params = cont_model_pars.copy()
    print(emcee_params)
    emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001),
                     max=np.log(2.0))
    result_emcee = cont_model.fit(data=spec.flux[cont_m], x=spec.dispersion[
        cont_m], params=emcee_params, method='emcee',
                             nan_policy='omit', fit_kws=emcee_kws)

    # emcee_params = line_model_pars.copy()
    # print(emcee_params)
    # emcee_params.add('__lnsigma', value=np.log(0.1), min=np.log(0.001),
    #                  max=np.log(2.0))
    # result_emcee = line_model.fit(data=spec.flux[line_m], x=spec.dispersion[
    #     line_m], params=emcee_params, method='emcee',
    #                          nan_policy='omit', fit_kws=emcee_kws)

    lmfit.report_fit(result_emcee)

    emcee_corner = corner.corner(result_emcee.flatchain,
                                 labels=result_emcee.var_names,
                                 truths=list(
                                     result_emcee.params.valuesdict().values()))
    plt.show()



def fit_and_resample_new(spec, foldername, redshift, pl_prefix,
                        line_prefixes, line_names, continuum_wavelengths, cosmology,
                        n_samples=100, cont_fit_weights=True,
                        line_fit_weights=True,
                        smooth = 10,
                        save_result_plots=True, resolution=None,
                        mgII_feII_prefix=None,
                        mgII_feII_int_range=None, verbosity=0):


    in_dict = load_qso_fit(foldername)

    if hasattr(in_dict['spec'], 'flux_err') == False:
        raise ValueError("The spectrum does not have usable flux errors")

    # # read original spectrum
    # spec = sod.SpecOneD()
    # spec.read_pypeit_fits(filename)

    cont_model_list = in_dict['cont_model_list']
    line_model_list = in_dict['line_model_list']
    cont_model_par_list = in_dict['cont_model_par_list']
    line_model_par_list = in_dict['line_model_par_list']

    cont_fit_z_flag = in_dict['cont_fit_z_flag']
    line_fit_z_flag = in_dict['line_fit_z_flag']

    # TODO getting z from line_model_par_list might cause problems,
    #  have general
    #  redshift parameter instead
    redsh_par = Parameters()
    redsh_par.add(line_model_par_list[0]['z'])
    redsh_par.value = redshift

    redsh_par_cont = Parameters()
    redsh_par_cont.add(line_model_par_list[0]['z'])
    redsh_par_cont.value = redshift

    result_example_dict = analyse(spec, cosmology, cont_model_list,
                                  line_model_list, cont_model_par_list,
                                  line_model_par_list, pl_prefix, line_names,
                                  line_prefixes, redshift,
                                  resolution=resolution,
                                  continuum_wavelengths=continuum_wavelengths,
                                  calc_bh_masses=True,
                                  mgII_feII_prefix=mgII_feII_prefix,
                                  mgII_feII_int_range=mgII_feII_int_range,
                                  verbosity=verbosity)

    n_results = len(result_example_dict)

    # 0) Set up the storage arrays
    # wav_array = np.array([3000, 1350, 1450, 3000])

    result_array = np.zeros(shape=(n_samples, n_results))

    # 1) Build the contiuum and line models with the pre-existing parameters
    # as initial conditions
    cont_model, cont_model_pars = build_continuum_model(cont_model_list,
                                                        cont_model_par_list,
                                                        cont_fit_z_flag,
                                                        redsh_par_cont,
                                                        z_vary=False)

    line_model, line_model_pars = build_line_model(line_model_list,
                                                   line_model_par_list,
                                                   line_fit_z_flag,
                                                   redsh_par,
                                                   z_vary=True)

    # 2) Resample the spectrum n times using the flux_err as 1 sigma of
    # Gaussian distribution

    spec_array = np.zeros(shape=(n_samples, len(spec.dispersion)))

    for idx, flux in enumerate(spec.flux):
        flux_err = spec.flux_err[idx]
        samples_flux = np.random.normal(flux, flux_err, n_samples)

        spec_array[:, idx] = samples_flux

    cont_m = np.logical_and(in_dict["mask_list"][1], in_dict[
        "mask_list"][0])
    line_m = np.logical_and(in_dict["mask_list"][2], in_dict[
        "mask_list"][0])
    m = in_dict["mask_list"][0]

    # Setup spectra and fits plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    # smooth spectrum
    spec.smooth(20, inplace=True)
    # ax.plot(spec.dispersion[m], spec.flux[m] * 1e+17, color='grey')
    print("SPECTRUM RAW", len(spec.raw_dispersion), len(spec.raw_flux),
          len(in_dict["mask_list"][0]))
    ax.plot(spec.raw_dispersion[m], spec.raw_flux[m] * 1e+17, color='grey')
    ax.plot(spec.dispersion[m], spec.flux[m] * 1e+17, color='k')

    # Begin loop
    for idx in range(n_samples):
        print(idx)
        spec_idx = spec_array[idx]

        idx_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=spec_idx,
                                flux_err=spec.flux_err, unit='f_lam')

        if smooth is not False:
            idx_spec.smooth(smooth, inplace=True)

        # 3 a) fit the continuum, then fit the lines -> record the results in
        # arrays

        in_dict['fit_with_weights'] = cont_fit_weights
        if in_dict['fit_with_weights']:
            cont_fit_result = cont_model.fit(idx_spec.flux[cont_m],
                                             cont_model_pars,
                                             x=idx_spec.dispersion[cont_m],
                                             weights=1./idx_spec.flux_err[
                                                 cont_m]**2)
        else:
            cont_fit_result = cont_model.fit(idx_spec.flux[cont_m],
                                             cont_model_pars,
                                             x=idx_spec.dispersion[cont_m])

        cont_fit_flux = cont_model.eval(cont_fit_result.params,
                                        x=spec.dispersion)

        res_flux = idx_spec.flux - cont_fit_flux

        in_dict['fit_with_weights'] = line_fit_weights
        if in_dict['fit_with_weights']:
            line_fit_result = line_model.fit(res_flux[line_m],
                                             line_model_pars,
                                             x=idx_spec.dispersion[line_m],
                                             weights=1./idx_spec.flux_err[
                                                 line_m]**2)
        else:
            line_fit_result = line_model.fit(res_flux[line_m],
                                             line_model_pars,
                                             x=idx_spec.dispersion[line_m])

        line_fit_flux = line_model.eval(line_fit_result.params,
                                        x=spec.dispersion)

        # Plot spectrum, continuum fit, and line fit for each draw
        ax.plot(spec.dispersion, cont_fit_flux*1e+17, color=dblue,
        alpha=0.5)
        ax.plot(spec.dispersion, (cont_fit_flux+line_fit_flux)*1e+17,
                color=vermillion, alpha=0.5)

        # 4) Evaluate the fit
        # a) continuum evaluation


        dummy_line_par_list = line_model_par_list.copy()
        dummy_cont_par_list = cont_model_par_list.copy()

        dummy_line_par_list = upd_param_values_from_fit(line_model_list,
                                                        dummy_line_par_list,
                                                        line_fit_result)
        dummy_cont_par_list = upd_param_values_from_fit(cont_model_list,
                                                        dummy_cont_par_list,
                                                        cont_fit_result)


        result_dict = analyse(spec, cosmology, cont_model_list,
                                  line_model_list, dummy_cont_par_list,
                                  dummy_line_par_list, pl_prefix, line_names,
                                  line_prefixes, redshift,
                                  resolution=resolution,
                                  continuum_wavelengths=continuum_wavelengths,
                                  calc_bh_masses=True,
                                  mgII_feII_prefix=mgII_feII_prefix,
                                  mgII_feII_int_range=mgII_feII_int_range,
                                  verbosity=verbosity)

        result_array[idx, :] = np.array(list(result_dict.values()))


    # End plotting routine for spectra
    plt.ylim(0.01, 1.3)
    plt.savefig(foldername + '/resampled_fits.pdf')


    raw_df = pd.DataFrame(data=result_array, columns=result_dict.keys())

    raw_df.to_hdf(foldername+'/resampled_fitting_results'+str(
        n_samples)+'_raw.hdf5', 'data')

    # 6) Evaluate raw resampled fitting results and save median/+-1sigma
    # fitting results
    stat_result_array = np.zeros(shape=(3, n_results))

    stat_result_array[0, :] = np.nanpercentile(raw_df.values, 50, axis=0)
    stat_result_array[1, :] = np.nanpercentile(raw_df.values, 13.6, axis=0)
    stat_result_array[2, :] = np.nanpercentile(raw_df.values, 86.4, axis=0)

    result_df = pd.DataFrame(data=stat_result_array, index=['median','lower',
                                                            'upper'],
                             columns=result_dict.keys())

    result_df.to_hdf(foldername + '/resampled_fitting_results' + str(
        n_samples) + '.hdf5', 'data')
    result_df.to_csv(foldername + '/resampled_fitting_results' + str(
        n_samples) + '.csv')

    # 7) Plot fitting results in one page
    if save_result_plots:
        for col in result_df.columns:

            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)

            median = result_df.loc['median', col]
            lower = result_df.loc['lower', col]
            upper = result_df.loc['upper', col]

            ax.hist(raw_df.loc[:, col], bins=round(n_samples / 2.))
            ax.axvline(median, c='k', ls='--', lw=2)
            ax.axvline(lower, c='k', ls='-.', lw=2)
            ax.axvline(upper, c='k', ls='-.', lw=2)

            med = '{:.4e}'.format(median)
            dp = '{:+.4e}'.format((lower - median))
            dm = '{:+.4e}'.format((upper - median))

            ax.set_title(med + ' ' + dp + ' ' + dm)
            ax.set_xlabel(r'{}'.format(col),
                           fontsize=15)

            plt.savefig(foldername +'/{}_results.pdf'.format(col))



def analyse_model(foldername,
                  cosmology,
                  pl_prefix,
                  line_names,
                  line_prefixes,
                  redshift,
                  resolution=None,
                  continuum_wavelengths=[3000, 1450, 1350, 2100, 5100],
                  calc_bh_masses=True,
                  mgII_feII_prefix=None,
                  mgII_feII_int_range=None,
                  verbosity=2):


    # read model from folder
    in_dict = load_qso_fit(foldername)

    if hasattr(in_dict['spec'], 'flux_err') == False:
        raise ValueError("The spectrum does not have usable flux errors")

    spec = in_dict['spec']

    cont_model_list = in_dict['cont_model_list']
    line_model_list = in_dict['line_model_list']
    cont_model_par_list = in_dict['cont_model_par_list']
    line_model_par_list = in_dict['line_model_par_list']

    print(in_dict['line_fit_result'].fit_report())
    print(in_dict['cont_fit_result'].fit_report())

    return analyse(spec, cosmology,
                   cont_model_list, line_model_list,
                   cont_model_par_list, line_model_par_list,
                   pl_prefix, line_names, line_prefixes, redshift,
                   resolution=resolution,
                   continuum_wavelengths=continuum_wavelengths,
                   calc_bh_masses=calc_bh_masses,
                   mgII_feII_prefix=mgII_feII_prefix,
                   mgII_feII_int_range=mgII_feII_int_range,
                   verbosity=verbosity)




def analyse(spec,
            cosmology,
            cont_model_list,
            line_model_list,
            cont_model_par_list,
            line_model_par_list,
          pl_prefix,
          line_names,
          line_prefixes,
          redshift,
          resolution=None,
          continuum_wavelengths=[3000, 1400, 1450, 1350, 2100, 5100],
          calc_bh_masses=True,
          mgII_feII_prefix = None,
          mgII_feII_int_range = None,
          verbosity=2):


    result_dict = {}

    # Analyse continuum
    if verbosity > 0:
        print('[INFO] ------------------')
        print('[INFO] Continuum Analysis')
        print('[INFO] ------------------')

    if 3000 not in continuum_wavelengths:
        continuum_wavelengths.append(3000)
    if 1400 not in continuum_wavelengths:
        continuum_wavelengths.append(1400)

    cont_wav_3000_idx = continuum_wavelengths.index(3000)
    cont_wav_1400_idx = continuum_wavelengths.index(1400)
    continuum_wavelengths = np.array(continuum_wavelengths)
    cont_fwav = np.zeros(len(continuum_wavelengths))


    for idx, cont in enumerate(cont_model_list):

        if cont.prefix in pl_prefix:

            params = cont_model_par_list[idx]
            # ATTENTION CHECK WHETHER TO INTEGRATE BALMER CONTINUUM OR NOT IN
            # CONT MEASUREMENTS!!!

            cont_fwav = cont.eval(params, x=continuum_wavelengths * (1. +
                                                                      redshift))
            # print(cont_fwav)
            # retrieve power law slope and amplitude (assume anchored at 2500)
            pl_slope = params[cont.prefix+'slope'].value
            # pl_amp = params[cont.prefix + 'amp'].value
            # pl_z = params[cont.prefix + 'z'].value
            #
            # cont_fwav = specmod.power_law_at_2500A(continuum_wavelengths * (1.
            #                                                          +redshift),
            #                    pl_amp, pl_slope, pl_z)
            # print(cont_fwav)

    cont_Lwav = calc_Lwav_from_fwav(cont_fwav, redshift, cosmology)
    L_bol = calc_QSO_Lbol_from_L3000(cont_Lwav[cont_wav_3000_idx] *
                                     continuum_wavelengths[cont_wav_3000_idx])
    L_bol_Ne19_1400 = calc_QSO_Lbol_Ne19(cont_Lwav[cont_wav_1400_idx] *
                                     continuum_wavelengths[
                                         cont_wav_1400_idx], 1400)


    for idx, wave in enumerate(continuum_wavelengths):
        result_dict['cont_flux_{}'.format(wave)] = cont_fwav[idx]
        result_dict['cont_L_{}'.format(wave)] = cont_Lwav[idx]

    result_dict['pl_slope'] = pl_slope
    result_dict['L_bol'] = L_bol
    result_dict['L_bol_Ne19_1400'] = L_bol_Ne19_1400

    if verbosity > 1:
        print('[INFO] Continuum wavelenghts for flux measurements:')
        print('[INFO] {}'.format(continuum_wavelengths))
        print('[INFO] f_lambda for continuum wavelengths (erg/s/AA/cm^2)')
        print('[INFO] {}'.format(cont_fwav))
        print('[INFO] L_lambda for continuum wavelengths (erg/s/AA)')
        print('[INFO] {}'.format(cont_Lwav))
        print('[INFO] Power law slope: {:.2f}'.format(pl_slope))
        print('[INFO] Bolometric luminosity: {:} (erg/s)'.format(L_bol))
        print('[INFO] Bolometric luminosity (Ne19, 1400A): {:} (erg/s)'.format(
            L_bol_Ne19_1400))

    # - build Fe continuum model (only Fe emission)

    if mgII_feII_prefix is not None:
        # Analyse lines
        if verbosity > 0:
            print('[INFO] ------------------')
            print('[INFO] FeII Analysis')
            print('[INFO] ------------------')

        if mgII_feII_int_range is None:
            mgII_feII_int_range = [2200, 3090]
            if verbosity > 0:
                print('[INFO] Integrating FeII emission over {} to {} AA '
                      '(default)'.format(mgII_feII_int_range[0],
                                         mgII_feII_int_range[1]))

        else:
            if verbosity > 0:
                print('[INFO] Integrating FeII emission over {} to {} AA '
                      '(user specified)'.format(mgII_feII_int_range[0],
                                         mgII_feII_int_range[1]))

        mgII_feII_int_range_z = [mgII_feII_int_range[0] * (redshift + 1),
                                 mgII_feII_int_range[1] * (redshift + 1)]

        # Construct the FeII model in the dispersion range requested for
        # analysis
        dispersion = np.arange(mgII_feII_int_range_z[0]-100,
                               mgII_feII_int_range_z[1]+100,
                               0.1)
        fe_flux = build_line_flux_from_line_models(dispersion,
                                                     mgII_feII_prefix,
                                                     cont_model_list,
                                                     cont_model_par_list)

        fe_flux_3000 = build_line_flux_from_line_models(np.array([3000* (redshift + 1)]),
                                                     mgII_feII_prefix,
                                                     cont_model_list,
                                                     cont_model_par_list)

        fe_fl = calc_integrated_flux(dispersion, fe_flux,
                                     range=mgII_feII_int_range_z)

        fe_L = calc_integrated_line_luminosity_new(dispersion,
                                                 fe_flux,
                                                 redshift,
                                                 cosmology,
                                                 range=mgII_feII_int_range)

        result_dict['FeII_L'] = fe_L
        result_dict['FeII_flux'] = fe_fl

        if verbosity > 0:
            print('[INFO] FeII luminosity: {:.2e}'.format(fe_L))
            print('[INFO] FeII flux: {:.2e}'.format(fe_fl))


        # Calculate bolometric luminosity according to Netzer 2019
        f_wav_3000 = fe_flux_3000[0] + cont_fwav[cont_wav_3000_idx]
        L_wav_3000 = calc_Lwav_from_fwav(f_wav_3000, redshift, cosmology)
        L_bol_Ne19 = calc_QSO_Lbol_Ne19(L_wav_3000*3000, 3000)

        result_dict['L_bol_Ne19_3000'] = L_bol_Ne19
        if verbosity > 1:
            print('[INFO] Bolometric luminosity (Ne19, 3000A): {:} (erg/s)'.format(
                L_bol_Ne19))


    # Analyse lines
    if verbosity > 0:
        print('[INFO] ------------------')
        print('[INFO] Line Analysis')
        print('[INFO] ------------------')

    dispersion = spec.dispersion


    # build the full continuum flux model including FeII and Balmer continuum
    cont_flux = build_cont_flux_from_cont_models(dispersion,
                                                 cont_model_list,
                                                 cont_model_par_list)

    for cont_params in cont_model_par_list:
        # print(cont_params)
        for param in cont_params:

            result_dict[param] = cont_params[param].value

    for line_params in line_model_par_list:
        # print(line_params)
        for param in line_params:

            result_dict[param] = line_params[param].value



    for idx, name in enumerate(line_names):

        line_prefix_list = line_prefixes[idx]

        line_flux = build_line_flux_from_line_models(dispersion,
                                                         line_prefix_list,
                                                         line_model_list,
                                                         line_model_par_list)

        # Calculate FWHM from line
        fwhm = calc_fwhm_from_line(dispersion, line_flux)

        if resolution is not None and fwhm[0] is not np.NaN:
            if verbosity > 1:
                print('[INFO] Correcting FWHM for provided resolution.')
            fwhm = np.sqrt(fwhm**2-resolution**2)


        flux_line = calc_integrated_flux(dispersion, line_flux)

        # if mgII_feII_prefix is not None:
        #     flux_line_FeII = calc_integrated_flux(dispersion, line_flux,
        #                                           range=mgII_feII_int_range_z)


        # get central wavelength from model (a bit complicated)
        for idx, line in enumerate(line_model_list):
            if line.prefix in line_prefix_list:
                try:
                    wav_rest = line_model_par_list[idx][line.prefix+'cen'].value
                except:
                    wav_rest = line_model_par_list[idx][line.prefix+'cen'].value


        # calculate peak from composite lines
        index_max = np.argmax(line_flux)
        wav_max = dispersion[index_max]
        line_z = (wav_max/wav_rest) - 1.
        # calculate redshift from composite lines

        L_line = calc_integrated_line_luminosity(dispersion,
                                                 line_flux,
                                                 redshift,
                                                 cosmology)

        ew = calc_equivalent_width_from_line_models(dispersion,
                                                    cont_flux,
                                                    line_flux,
                                                    redshift=redshift)

        # calculate blueshift compared to given rest_wavelength
        dv = lu.dv_from_z(line_z, redshift, rel=True).value

        result_dict[name+'_wav_cen'] = wav_max
        result_dict[name+'_z_cen'] = line_z
        result_dict[name+'_vshift'] = dv
        result_dict[name+'_fwhm'] = fwhm[0]
        result_dict[name+'_EW'] = ew
        result_dict[name+'_L'] = L_line
        result_dict[name + '_flux'] = flux_line

        # Apply FWHM correction for CIV line
        if name == 'CIV':
            fwhm_corr = correct_CIV_fwhm(fwhm[0], -dv)
            result_dict[name + '_fwhm_corr'] = fwhm_corr

            if resolution is not None and fwhm_corr is not np.NaN:
                if verbosity > 1:
                    print('[INFO] Correcting  corrected CIV FWHM for provided '
                          'resolution.')
                fwhm_corr = np.sqrt(fwhm_corr ** 2 - resolution ** 2)

        if verbosity > 1:
            print('[INFO] Line analysis for {}'.format(name))
            print('[INFO] {} rest frame wavelength is {}'.format(name,
                                                                 wav_rest))
            print('[INFO] {:} line central wavelength: {}'.format(
                name, wav_max))
            print('[INFO] {} line redshift: {}'.format(name, line_z))
            print('[INFO] {} line velocity shift: {}'.format(name, dv))
            print('[INFO] {:} line FWHM: {:.4e}'.format(name, fwhm[0]))
            print('[INFO] {:} line EW: {:.4e}'.format(name, ew))
            print('[INFO] {:} line flux: {:.4e}'.format(name, flux_line))
            print('[INFO] {:} line luminosity: {:.4e}'.format(name, L_line))

            if name == 'CIV':
                print('[INFO] {:} line corrected FWHM: {:.4e}'.format(name,
                                                                  fwhm_corr))


        # Black hole mass calculation
        if calc_bh_masses:

            for idx, wave in enumerate(continuum_wavelengths):

                bhmass, ref = calc_BH_masses(result_dict['cont_L_{'
                                                               '}'.format(wave)],
                                                   wave,
                                                   fwhm,
                                                   name,
                                                   verbosity=verbosity)

                if bhmass is not None and not np.any(np.isnan(bhmass)) and not \
                        isinstance(bhmass, list):

                    bhmass = bhmass[0]

                    bhmass_name = name + '_BHmass_' + str(wave) + '_' + ref
                    result_dict[bhmass_name] = bhmass


                    edd_lum = calc_Edd_luminosity(bhmass)
                    edd_lum_name = name + '_EddLumR_' + str(wave) + '_' + ref
                    result_dict[edd_lum_name] = L_bol / edd_lum

                    if verbosity > 1:
                        print('[INFO] Black hole mass {}: {}'.format(
                            bhmass_name, bhmass))
                        print('[INFO] Eddington luminosity ratio {}: {}'.format(
                            edd_lum_name, L_bol / edd_lum))

                elif bhmass is not None and not np.any(np.isnan(bhmass)) and \
                        isinstance(bhmass, list):

                    for idx, bhm in enumerate(bhmass):
                        reference = ref[idx]
                        bhm = bhm[0]

                        bhmass_name = name + '_BHmass_' + str(wave) + '_' + reference
                        result_dict[bhmass_name] = bhm

                        edd_lum = calc_Edd_luminosity(bhm)
                        edd_lum_name = name + '_EddLumR_' + str(
                            wave) + '_' + reference
                        result_dict[edd_lum_name] = L_bol / edd_lum

                        if verbosity > 1:
                            print('[INFO] Black hole mass {}: {}'.format(
                                bhmass_name, bhm))
                            print(
                                '[INFO] Eddington luminosity ratio {}: {}'.format(
                                    edd_lum_name, L_bol / edd_lum))

                elif bhmass is not None and np.any(np.isnan(bhmass)):
                    bhmass_name = name + '_BHmass_' + str(
                        wave) + '_' + ref
                    edd_lum_name = name + '_EddLumR_' + str(
                        wave) + '_' + ref
                    result_dict[bhmass_name] = np.NaN
                    result_dict[edd_lum_name] = np.NaN

            # NEEDS TO BE TESTED!!!!
            if name == 'Halpha':

                bhmass, ref = calc_Halpha_BH_mass(L_line, fwhm[0])

                bhmass_name = name + '_BHmass_' + str(wave) + '_' + ref
                result_dict[bhmass_name] = bhmass

                edd_lum = calc_Edd_luminosity(bhmass)
                edd_lum_name = name + '_EddLumR_' + str(wave) + '_' + ref
                result_dict[edd_lum_name] = L_bol / edd_lum

                if verbosity > 1:
                    print('[INFO] Black hole mass {}: {}'.format(
                        bhmass_name, bhmass))
                    print('[INFO] Eddington luminosity ratio {}: {}'.format(
                        edd_lum_name, L_bol / edd_lum))



        # Calculate BH mass for corrected CIV fwhm
        if name == 'CIV':
            wave = 1350
            L_wav = result_dict['cont_L_{}'.format('1350')]
            bhmass_corr, ref = calc_CIV_BHmass_Co17(L_wav, wave, fwhm_corr)
            bhmass_name = name + '_BHmass_corr_' + str(wave) + '_' + ref

            result_dict[bhmass_name] = bhmass_corr

            edd_lum = calc_Edd_luminosity(bhmass_corr)
            edd_lum_name = name + '_EddLumR_corr_' + str(
                wave) + '_' + ref

            result_dict[edd_lum_name] = L_bol / edd_lum

            if verbosity > 1:
                print('[INFO] CIV Black hole mass (Co17) {}: {}'.format(
                    bhmass_name, bhmass_corr))
                print(
                    '[INFO] CIV Eddington luminosity ratio (Co17) {}: {'
                    '}'.format(
                        edd_lum_name, L_bol / edd_lum))


    # Calculate MgII over FeII if available
    if 'FeII_flux' in result_dict and 'MgII_flux' in result_dict:
        if verbosity > 0:
            print('[INFO] ------------------')
            print('[INFO] FeII/MgII ratio analysis')
            print('[INFO] ------------------')

        feII_over_mgII = result_dict['FeII_flux'] / result_dict['MgII_flux']
        result_dict['FeII_over_MgII'] = feII_over_mgII
        feII_over_mgII_L = result_dict['FeII_L'] / result_dict['MgII_L']
        result_dict['FeII_over_MgII_L'] = feII_over_mgII_L
        if verbosity > 1:
            print('[INFO] MgII/FeII flux ratio: {:.2e}'.format(
                feII_over_mgII))
            print('[INFO] MgII/FeII luminosity ratio: {:.2e}'.format(
                feII_over_mgII_L))


    # Calculate monochromatic magnitudes based on fit

    # DM = -cosmology.distance_modulus(redshift)
    # K = cosmology.k_correction(redshift, -0.5)
    #
    #
    # print("Corresponding monochromatic absolute magnitudes: ",
    #       calc_abmag_from_fwav(f_wav, wav_arr) + DM + K)


    return result_dict



# ------------------------------------------------------------------------------
# Test and plots
# ------------------------------------------------------------------------------

def plot_qso_fit(in_dict, plot_regions):

    # Tex font
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    fig = plt.figure(num=None, figsize=(8, 4), dpi=140)
    fig.subplots_adjust(left=0.16, bottom=0.14, right=0.96, top=0.95, wspace=0)

    spec = in_dict['spec']

    spec = spec.smooth(5)

    cont = in_dict['cont_fit_spec']
    line = in_dict['line_fit_spec']

    print(spec.dispersion, spec.flux, spec.mask)

    n = len(plot_regions)

    print(n)

    for idx in range(n):
        ax = fig.add_subplot(1, n, idx+1)

        ax.plot(spec.dispersion[spec.mask], spec.flux[spec.mask], 'k')
        ax.plot(cont.dispersion, cont.flux, color=blue)
        ax.plot(line.dispersion, cont.flux + line.flux, color=vermillion)

        ax.set_xlim(plot_regions[idx][0], plot_regions[idx][1])

    plt.show()



def make_line_fit_plots(in_dict, cont_prefix, fe_prefixes,
                        line_prefixes, line_name, redsh=None, plot_width=10000):

    # Tex font
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    spec = in_dict['spec']
    mask = in_dict['mask_list'][0]

    if line_name == "MgII":
        cen = 2798
    elif line_name == "CIV":
        cen = 1549
    elif line_name == "Hbeta":
        cen = 4861
    else:
        raise ValueError('Line name not recognized. Supported line names: '
                         'MgII, CIV, Hbeta')

    if redsh is None:
        redsh = in_dict['line_model_par_list'][0]['z'].value
        print(redsh)
    else:
        redsh = 0

    cen *= (redsh + 1.)

    width = plot_width / c_km_s * cen

    # z0_dispersion = spec.dispersion / (redsh + 1)

    fig = plt.figure(num=None, figsize=(4, 4), dpi=140)
    fig.subplots_adjust(left=0.16, bottom=0.14, right=0.96, top=0.95)
    ax1 = fig.add_subplot(1, 1, 1)

    ax1_twinx = ax1.twinx()

    ax1.plot(spec.dispersion[mask], spec.flux[mask], color='k')

    # Plot regions fitted (masked in mask 2)
    fit_mask = in_dict['mask_list'][2]


    trans = mtransforms.blended_transform_factory(
        ax1.transData, ax1.transAxes)
    spec2 = in_dict['spec'].copy()
    spec2.flux[np.invert(fit_mask)] = np.NaN

    ax1.fill_between(spec2.dispersion, 0, 1, where=spec2.flux > 0,
                              facecolor='grey', alpha=0.2,
                              transform=trans, zorder=0)




    # z0spec = spec.copy()
    # z0spec.dispersion = z0_dispersion
    y_max = np.max(spec.trim_dispersion([cen-width/2., cen+width/2.]).flux)
    y_min = np.min(spec.trim_dispersion([cen - width / 2., cen + width /
                                        2.]).flux)


    cont_flux = np.zeros(len(spec.dispersion))
    fe_flux = np.zeros(len(spec.dispersion))

    for idx, model in enumerate(in_dict['cont_model_list']):
        print(model, model.prefix, cont_prefix)
        if model.prefix == cont_prefix:
            params = in_dict['cont_model_par_list'][idx]
            print(params)
            cont_flux = model.eval(params, x=spec.dispersion)

        if model.prefix in fe_prefixes:
            params = in_dict['cont_model_par_list'][idx]
            fe_flux += model.eval(params, x=spec.dispersion)

    print(cont_flux, spec.dispersion)

    line_flux = np.zeros(len(spec.dispersion))

    for idx, model in enumerate(in_dict['line_model_list']):
        params = in_dict['line_model_par_list'][idx]
        line_flux += model.eval(params, x=spec.dispersion)
        line = model.eval(params, x=spec.dispersion)

        em, = ax1_twinx.plot(spec.dispersion, line + max(y_max*0.1,y_min*0.6),
                       color='grey', label = r'$\rm{Emission\ line\ '
                                             r'components}$')

    pl, = ax1.plot(spec.dispersion, cont_flux, color=blue, label=r'$\rm{'
                                                                 r'Power}\ '
                                                               r'\rm{law}\ '
                                                               r'\rm{'
                                                               r'cont.}$',
                   lw=2.5)
    fe, = ax1_twinx.plot(spec.dispersion, fe_flux + max(y_min*0.6,y_max*0.1),
                         color =
    green,
                    label=r'$\rm{Iron\ emission}$', lw=2.5)
    fit, = ax1.plot(spec.dispersion, line_flux + cont_flux + fe_flux,
             color=vermillion, label=r'$\rm{Full\ fit}$', lw=2.5)


    ax1.set_xlim(cen-width/2.,
                 cen+width/2.)
    ax1_twinx.set_xlim(cen-width/2.,
                       cen+width/2.)



    ax1.set_ylim(max(y_min*0.5,0), y_max*1.35)
    ax1_twinx.set_ylim(max(y_min*0.5,0), y_max*1.35)

    ax1_twinx.set_yticks([])

    ax1.set_xlabel(r'$\rm{Observed\ Wavelength}\ [\rm{\AA}]$', fontsize=15)
    ax1.set_ylabel(r'$\rm{Flux}\ f_{\lambda}\ [\rm{erg}\,\rm{s}^{-1}\,'
                   r'\rm{cm}^{-2}\,\rm{\AA}^{-1}]$',
        fontsize=15)

    ax1.legend(handles=[fit, pl, fe, em], fontsize=10, loc=0)

    plt.show()


def test_BH_mass_calc_VP06VO09():
    # test with some values... not meant for a reality check,
    # just to see the function output
    fwhm = 3500

    wav = 1350

    L_wav = 10**46

    line = "MgII"

    print('MgII: ', calc_BHmass_VP06VO09(L_wav, wav, fwhm, line=line)/1e+9)

    line = "CIV"

    print('CIV: ', calc_BHmass_VP06VO09(L_wav, wav, fwhm, line=line)/1e+9)

    wav = 5100

    line = "Hbeta"

    print('Hbeta: ', calc_BHmass_VP06VO09(L_wav, wav, fwhm, line=line)/1e+9)



def plot_optnir_fit(foldername, cont_prefix, fe_prefixes,
                    line_prefixes, panels=3, wavel=[[5400,
                                                                      6800],[10000,
                                                                    13500],
                                             [15000,23200]], redsh=None):
    # Tex font
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)

    in_dict = load_qso_fit(foldername)
    spec = in_dict['spec']
    mask = in_dict['mask_list'][0]

    cont_mask = in_dict['mask_list'][1]
    line_mask = in_dict['mask_list'][2]

    if redsh is None:
        redsh = in_dict['line_model_par_list'][0]['z'].value
        print(redsh)
    else:
        redsh = 0

    fig = plt.figure(figsize=(12, 9))

    #test if wavel leng is the same as panels

    for idx in range(panels):
        ax = fig.add_subplot(panels, 1, idx+1)
        # res_ax = fig.add_subplot(2*panels, 1, 2*idx+2)

        def forward(x):
            return x / (1. + redsh)

        def inverse(x):
            return x * (1. + redsh)

        ax_main_rest = ax.secondary_xaxis('top',functions=(forward, inverse))

        if idx == 0:
            ax_main_rest.set_xlabel(r'$\rm{Rest\ frame\ wavelength\ [\mbox{'
                                    r'\normalfont\AA}]}$', fontsize=15)
        if idx == panels-1:
            ax.set_xlabel(r'$\rm{Observed\ frame\ wavelength\ [\mbox{'
                                    r'\normalfont\AA}]}$', fontsize=15)

        # twinx = ax.twinx()

        print(wavel[idx])
        trimmed = spec.trim_dispersion([wavel[idx][0], wavel[idx][1]])
        y_max = np.max(trimmed.flux[trimmed.mask])
        # y_min = np.min(
        #     spec.trim_dispersion([wavel[idx][0], wavel[idx][1]]).flux[mask])



        cont_flux = np.zeros(len(spec.dispersion))
        fe_flux = np.zeros(len(spec.dispersion))

        for jdx, model in enumerate(in_dict['cont_model_list']):
            print(model, model.prefix, cont_prefix)
            if model.prefix == cont_prefix:
                params = in_dict['cont_model_par_list'][jdx]

                cont_flux = model.eval(params, x=spec.dispersion)

            if model.prefix in fe_prefixes:
                params = in_dict['cont_model_par_list'][jdx]
                fe_flux += model.eval(params, x=spec.dispersion)

        line_flux = np.zeros(len(spec.dispersion))

        for jdx, model in enumerate(in_dict['line_model_list']):
            params = in_dict['line_model_par_list'][jdx]
            line_flux += model.eval(params, x=spec.dispersion)
            line = model.eval(params, x=spec.dispersion)


        flux = np.copy(spec.flux)
        flux_err = np.copy(spec.flux_err)

        for y in [flux, flux_err]:

            y[np.invert(mask)] = np.NaN


        ax.plot(spec.dispersion, flux, 'k')
        ax.plot(spec.dispersion, flux_err, 'grey')

            #
            # em, = ax.plot(spec.dispersion[linemask],
            #                      line[linemask],
            #                      color=purple, label=r'$\rm{Emission\ line\ '
            #                                          r'components}$')

        pl, = ax.plot(spec.dispersion, cont_flux, color=dblue, label=r'$\rm{'
                                                                     r'Power}\ '
                                                                     r'\rm{law}\ '
                                                                     r'\rm{'
                                                                     r'cont.}$',
                       lw=1.5)

        femask = fe_flux < 1000
        fe, = ax.plot(spec.dispersion[femask],
                             fe_flux[femask],
                             color=
                             green,
                             label=r'$\rm{Iron\ emission}$', lw=1.5)

        linemask = line_flux < 1000
        line, = ax.plot(spec.dispersion[linemask], line_flux[linemask],
                        color=purple, label=r'$\rm{Emission\ line\ '
                                                     r'components}$', lw=1.5)

        fit, = ax.plot(spec.dispersion, line_flux + cont_flux + fe_flux,
                        color=vermillion, label=r'$\rm{Full\ fit}$', lw=2)

        trans = mtransforms.blended_transform_factory(
            ax.transData, ax.transAxes)

        yy = in_dict['spec'].copy()
        yy.flux[np.invert(cont_mask)] = np.NaN

        ax.fill_between(yy.dispersion, 0, 1, where=yy.flux>0,
                        facecolor=green, alpha=0.2, transform=trans)

        yy = in_dict['spec'].copy()
        yy.flux[np.invert(line_mask)] = np.NaN

        ax.fill_between(yy.dispersion, 0, 1, where=yy.flux > 0,
                        facecolor=vermillion, alpha=0.2, transform=trans)




        ax.set_xlim(wavel[idx])

        ax.set_ylim(-1e-17, y_max)



    plt.show()


def test_fit_and_resample():

    cos = cosm.Cosmology()
    cos.set_zentner2007_cosmology()

    foldername = 'J2125-1719_hbeta'
    redsh = 3.906
    pl_prefix = ['pl_']
    fe_prefixes = []
    line_prefixes = [['hbeta_b_', 'hbeta_n_']]
    line_names = ['Hbeta']
    line_wavs = [5100]



    fit_and_resample(foldername, redsh, pl_prefix, fe_prefixes,
                     line_prefixes, line_names, line_wavs, cos, n_samples=1000)





# test_BH_mass_calc_VP06VO09()

# cos = cosm.Cosmology()
#
# cos.set_zentner2007_cosmology()
#
#
#
# line_prefixes = [['mgII_b_', 'mgII_n_'], ['hbeta_b_', 'hbeta_n_']]
# line_names = ['MgII', 'Hbeta']
#
# # line_prefixes = [['hbeta_b_', 'hbeta_n_']]
# # line_names = ['Hbeta']
#
# pl_prefix = ['pl_']
# fe_prefixes = []
# redsh = 3.69
#
# # redsh = 6.3637
#
# analyse_qso_fit('J0341_fit', redsh, pl_prefix, fe_prefixes,
#                 line_prefixes, line_names, cos)


# in_dict = load_qso_fit('J0341_fit')
# in_dict = load_qso_fit('J2125-1719_hbeta')
# cont_prefix = 'pl_'
# # fe_prefixes = ['fe_']
#
# make_line_fit_plots(in_dict, cont_prefix, fe_prefixes, line_prefixes, 'Hbeta',
#                     redsh, width = 10000)

# plot_ranges = [[4000,7600],[10300,13300],[20000,23200]]
# plot_qso_fit(in_dict, plot_ranges)

# test_fit_and_resample()


#
# def build_full_model(in_dict):
#
#     # add all continuum model together
#     # add all emission line models on top
#     #
#
#     pass
#
# def spec_lmfit_residual(spectrum, model, params):
#
#
#
#     pass
#
# def spec_lnprob():
#
#     return -0.5 * np.sum((residual_DPL(p) / sigma_arr_z)**2 + np.log(2 * np.pi * sigma_arr_z**2))
#
#
#
# def fit_mcmc():
#
#     p3 = lmfit.Parameters()
#     p3.add_many(('phi_star', 1e-6, True, 1e-9, 1e-6),
#                ('M_star', -28, True, -30, -25),
#                ('alpha', -2, True, -3, -0),
#                ('beta', -3, True, -6, -2.5),
#                )
#
#     mini3 = lmfit.Minimizer(lnprob_DPL, p3)
#
#     res3 = mini3.emcee(burn=300, nwalkers = 100, steps=10000, params=p3, seed=23)
#
#     pass
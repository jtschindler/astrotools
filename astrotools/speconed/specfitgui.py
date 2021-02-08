
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

from astrotools.speconed.specfit_models import *


from .speconed import datadir


black = (0, 0, 0)
orange = (230/255., 159/255., 0)
blue = (86/255., 180/255., 233/255.)
green = (0, 158/255., 115/255.)
yellow = (240/255., 228/255., 66/255.)
dblue = (0, 114/255., 178/255.)
vermillion = (213/255., 94/255., 0)
purple = (204/255., 121/255., 167/255.)


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


# USABILIY Implementation:

# 2) Turn vary off for all parameters in specific cont/line model

# SCIENCE Implementation:
# 1) Balmer continuum model
# 3) Check FE templates and implement useful ones, add useful parameters
# 4) Rebinning versus interpolation???
# 5) Add flux error handling, use Eduardo's spectrum to test this
# 6) Add, 1000 refit error estimation AND MCMC fitting!

class SpecFitCanvas(FigureCanvas):

    """A FigureCanvas for plotting one spectrum as a result of an operation.

    This class provides the plotting routines for plotting a spectrum
    resulting from a multiplication of division of two spectra.
    """

    def __init__(self, parent=None, in_dict=None):

        """__init__ method for the SpecFitCanvas class

        Parameters
        ----------
        parent : obj, optional
            Parent class of SpecFitCanvas
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data.
        """

        fig = plt.figure(constrained_layout=True)

        gs = gridspec.GridSpec(4, 1)

        # main axis
        self.ax_main = fig.add_subplot(gs[:2,0])
        # axis to display the continuum subtracted spectrum
        self.ax_cont = fig.add_subplot(gs[2:4,0], sharex= self.ax_main)
        # axis to display the residual of the continuum/line subtraction
        # self.ax_resd = fig.add_subplot(gs[3,0], sharex= self.ax_main)
        # a twin axis of the main axis to plot the masked regions
        self.ax_main_twinx = self.ax_main.twinx()




        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


        self.specfit_plot(in_dict)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def specfit_plot(self, in_dict):

        """Plot the spectra on the canvas

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        """

        # clear all axes
        self.ax_main.clear()
        self.ax_main_twinx.clear()
        self.ax_cont.clear()
        # self.ax_resd.clear()

        # Plot the main spectrum
        self.plot_spec(in_dict)

        # Set the plot boundaries
        trimmed_spec = in_dict['spec'].trim_dispersion([in_dict['x_lo'],
                                                        in_dict['x_hi']])

        ylim_min, ylim_max = trimmed_spec.get_specplot_ylim()

        in_dict['y_lo'] = ylim_min
        in_dict['y_hi'] = ylim_max

        self.ax_main.set_xlim(in_dict['x_lo'], in_dict['x_hi'])
        self.ax_main.set_ylim(0, in_dict['y_hi'])


        # If fits are available show the continuum subtracted and residual
        # spectrum
        if in_dict['cont_fit_spec'] is not None:
            self.plot_cont(in_dict)
        # if in_dict['line_fit_spec'] is not None:
        #     self.plot_resd(in_dict)

        self.ax_main_twinx.set_ylim(0, 1)

        def forward(x):
            return x / (1. + in_dict['redshift'])

        def inverse(x):
            return x * (1. + in_dict['redshift'])

        self.ax_main_rest = self.ax_main.secondary_xaxis('top',
                                                         functions=(forward,
                                                                    inverse))
        self.ax_main_rest.set_xlabel(r"Rest frame wavelength [Angstroem]")
        self.ax_cont.set_xlabel(r"Observed frame wavelength [Angstroem]")

        self.draw()


    def plot_spec(self, in_dict):
        """Plot the main, continuum subtracted and full residual spectra

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        """

        spec = in_dict['spec'].copy()

        # Plot the spectrum
        if spec.flux_err is not None:
            self.ax_main.plot(spec.dispersion[spec.mask], spec.flux_err[spec.mask],
                          'grey')
        self.ax_main.plot(spec.dispersion[spec.mask], spec.flux[spec.mask], 'k')


        # Define the mask colors
        mask_colors = ['0.2', dblue, vermillion]

        # Plot the masks
        for idx, mask in enumerate(in_dict['mask_list']):
            if idx == 0:
                mask = np.invert(mask)

            trans = mtransforms.blended_transform_factory(
                self.ax_main.transData, self.ax_main.transAxes)
            spec = in_dict['spec'].copy()
            spec.flux[np.invert(mask)] = np.NaN

            self.ax_main.fill_between(spec.dispersion, 0, 1, where=spec.flux
                                                                   > 0,
                                      facecolor=mask_colors[idx], alpha=0.2,
                                      transform=trans)

        # Plot the continuum fit components
        for idx, model in enumerate(in_dict['cont_model_list']):

            params = in_dict['cont_model_par_list'][idx]
            x = spec.dispersion
            y = model.eval(params, x=x)

            self.ax_main.plot(x, y, color=green)

        # Plot the line fit components
        for idx, model in enumerate(in_dict['line_model_list']):
            params = in_dict['line_model_par_list'][idx]
            x = spec.dispersion
            y = model.eval(params, x=x)

            if in_dict['cont_fit_spec'] is not None:

                cont_fit = in_dict["cont_fit_spec"].copy()
                self.ax_main.plot(x, y+cont_fit.flux, color='grey')

            else:
                self.ax_main.plot(x, y, color='grey')

        # Plot the initial continuum model
        if in_dict["cont_init_spec"] is not None:
            init_fit = in_dict['cont_init_spec'].copy()

            self.ax_main.plot(init_fit.dispersion, init_fit.flux,
                              color=dblue, ls='--', alpha=0.5)

        # Plot the best-fit continuum model
        if in_dict["cont_fit_spec"] is not None:

            cont_fit= in_dict["cont_fit_spec"].copy()

            self.ax_main.plot(cont_fit.dispersion, cont_fit.flux, color=dblue)

        # Plot the sum of the best-fit continuum and line models
        if (in_dict["line_fit_spec"] is not None) and (in_dict["cont_fit_spec"] is not None):

            line_fit = in_dict['line_fit_spec'].copy()
            cont_fit = in_dict["cont_fit_spec"].copy()

            self.ax_main.plot(cont_fit.dispersion,
                              cont_fit.flux+line_fit.flux, color=vermillion)




    def plot_cont(self, in_dict):
        """Plot the continuum subtracted spectrum

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        """

        spec = in_dict['spec'].copy()

        cont_fit = in_dict['cont_fit_spec'].copy()

        # Calculated the continuum subtracted spectrum
        res = spec.subtract(cont_fit)

        # Plot the continuum subtracted spectrum
        self.ax_cont.plot(res.dispersion[res.mask], res.flux[res.mask], 'k')
        # Plot the 0 flux line
        self.ax_cont.plot(res.dispersion, res.dispersion*0, ls='--',
                          color='grey')

        # Set the plot boundaries
        # trimmed_spec = res.trim_dispersion(
            # [in_dict['x_lo'], in_dict['x_hi']])
        # y_lo = min(trimmed_spec.flux[trimmed_spec.mask])
        # y_hi = max(trimmed_spec.flux[trimmed_spec.mask])

        self.ax_cont.set_ylim(in_dict['y_lo'], in_dict['y_hi'])


        # Plot the initial line model
        if in_dict["line_init_spec"] is not None:
            init_fit = in_dict["line_init_spec"].copy()

            self.ax_cont.plot(init_fit.dispersion, init_fit.flux,
                              color=vermillion, ls='--',
                              alpha=0.5)

        # Plot the best fit line model
        if in_dict["line_fit_spec"] is not None:
            line_fit = in_dict["line_fit_spec"].copy()

            self.ax_cont.plot(line_fit.dispersion, line_fit.flux,
                              color=vermillion)

        # Plot the line fit components
        for idx, model in enumerate(in_dict['line_model_list']):
            params = in_dict['line_model_par_list'][idx]
            x = spec.dispersion
            y = model.eval(params, x=x)

            self.ax_cont.plot(x, y, color=green, ls='-.')


    def plot_resd(self, in_dict):
        """Plot the residual spectrum

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        """

        spec = in_dict['spec'].copy()
        cont_fit = in_dict['cont_fit_spec'].copy()
        line_fit = in_dict['line_fit_spec'].copy()

        # Calculate the residual spectrum
        res = spec.subtract(cont_fit).subtract(line_fit)

        # Plot the residual spectrum
        self.ax_resd.plot(res.dispersion[spec.mask], res.flux[spec.mask], 'k')
        # Plot the 0 flux line
        self.ax_resd.plot(res.dispersion, res.dispersion*0,ls='--', color='grey')
        # Set the plot boundaries
        # y_lo, y_hi = res.trim_dispersion([in_dict['x_lo'],in_dict[
            # 'x_hi']]).get_specplot_ylim()
        y_lo = min(res.trim_dispersion([in_dict['x_lo'],in_dict['x_hi']]).flux)
        y_hi = max(res.trim_dispersion([in_dict['x_lo'],in_dict['x_hi']]).flux)

        self.ax_resd.set_ylim(y_lo, y_hi)


class SpecFitGui(QMainWindow):
    """The interactive SpecFit GUI.

    This class provides all interactive capabilities for the fitting of
    one dimensional spectra (SpecOneD objects).

    Attributes
    ----------

    """

    def __init__(self, spec, redshift=None):
        """__init__ method for the SpecFitGui class

        Parameters
        ----------
        spec : SpecOneD
            A SpecOneD 1D spectrum to fit
        """

        QtWidgets.QMainWindow.__init__(self)

        # Initialize the mask arrays

        # Region NOT considered for any fit
        # self.mask_out = np.ones(spec.dispersion.shape, dtype=bool)
        self.mask_out = spec.mask
        # Region ONLY considered for continuum fit, except "mask_out" parts
        self.mask_in_contin = np.zeros(spec.dispersion.shape, dtype=bool)
        # Region ONLY considered for emission line fit, except "mask_out" parts
        self.mask_in_emline = np.zeros(spec.dispersion.shape, dtype=bool)

        # Calculate some general flux properties for first guess
        # values in the continuum fit
        if redshift is not None:
            self.redshift = redshift
            self.flux_2500 = np.mean(spec.trim_dispersion(
                                        [2400*(redshift+1.),
                                         2600*(redshift+1.)]).flux)
        else:
            self.redshift = 0.0


        # Initialize the input dictionary (in_dict) to store all important data
        self.in_dict = {'spec': spec,
                        'mask_list': [self.mask_out, self.mask_in_contin,  self.mask_in_emline],
                        'cont_fit_spec':None,
                        'cont_init_spec': None,
                        'line_fit_spec':None,
                        'line_init_spec':None,
                        'x_lo': None,
                        'x_hi': None,
                        'y_lo': None,
                        'y_hi': None,
                        'line_model_list': [],
                        'line_model_par_list': [],
                        'cont_model_list': [],
                        'cont_model_par_list': [],
                        'redshift':None}

        self.in_dict['redshift'] = self.redshift

        # Calculate default plot limits
        self.in_dict['x_lo'] = min(self.in_dict['spec'].dispersion)
        self.in_dict['x_hi'] = max(self.in_dict['spec'].dispersion)
        y_lo, y_hi = self.in_dict['spec'].get_specplot_ylim()

        self.in_dict['y_lo'] = y_lo
        self.in_dict['y_hi'] = y_hi
        # Set up dispersion zoom variables
        self._wx1 = None
        self._wx2 = None

        # Set up masking variables
        self.mx1 = 0
        self.mx2 = 0
        self.active_mask = 0

        # Set up continuum fitting variables
        self.cont_model_pars = Parameters()
        self.cont_fit_result = None
        self.cont_model = None
        self.cont_model_type = None
        self.cont_model_list = []
        self.cont_model_par_list = []
        self.cont_fit_z_flag = False

        # Set up line fitting variables
        self.line_fit_result = None
        self.line_model = None
        self.line_model_pars = None
        self.line_model_list = []
        self.line_model_par_list = []
        self.line_fit_z_flag = False

        self.fit_with_weights = True

        # REVIEW THIS
        # Set up the global redshift parameter
        self.redsh_par = Parameters()
        if redshift is not None:
            self.redsh_par.add('z', value=self.redshift, min=self.redshift*0.9,
                               max=self.redshift*1.1,
                               vary=True)
        else:
            self.redsh_par.add('z', value=0.0, min=0.0, max=1, vary=True)

        # Set up help window toggle flag
        self.help_window_flag = False

        # Set up reference functions for window functionality
        self.last_on_press = self.on_press_main
        self.last_help_label_function = self.main_help_labels

        # Add the main widget
        self.main_widget = QtWidgets.QWidget(self)

        # Layout of the main GUI
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        # # Set the lower and upper boundaries for the first plotting
        # self.in_dict['x_lo'] = min(self.in_dict['spec'].dispersion)
        # self.in_dict['x_hi'] = max(self.in_dict['spec'].dispersion)

        # Initialize the SpecFitCanvas
        self.specfitcanvas = SpecFitCanvas(self, in_dict=self.in_dict)
        self.specfitcanvas.setMinimumSize(600, 400)
        # Add the SpecFitCanvas to the SpecFitGui Layout
        self.plot_box = QHBoxLayout()
        self.plot_box.addWidget(self.specfitcanvas)
        self.layout.addLayout(self.plot_box)

        # Set the ClickFocus active on the SpecFitCanvas
        self.specfitcanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.specfitcanvas.setFocus()

        self.setCentralWidget(self.main_widget)

        self.mpl_toolbar = NavigationToolbar(self.specfitcanvas, self.main_widget)

        self.setWindowTitle("Interactive Spectral Fitting GUI")

        # Initialize general and current on press event ID's
        self.gcid = None
        self.cid = None

        # Initialize the Key Press Event for the SpecFitCanvas
        self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                    self.on_press_main)

    def closeEvent(self, event):
        """This function modifies the standard closing of the GUI

        Parameters
        ----------
        event : event
            The closing event.
        """

        result = QtWidgets.QMessageBox.question(self,
                      "Exit Dialog",
                      "Are you sure you want to exit ?",
                      QtWidgets.QMessageBox.Yes| QtWidgets.QMessageBox.No)
        event.ignore()

        if result == QtWidgets.QMessageBox.Yes:
            event.accept()

    def redraw(self):
        """ Redraw the canvas

        """
        self.in_dict['cont_model_list'] = self.cont_model_list
        self.in_dict['cont_model_par_list'] = self.cont_model_par_list
        self.in_dict['line_model_list'] = self.line_model_list
        self.in_dict['line_model_par_list'] = self.line_model_par_list
        self.in_dict['redshift'] = self.redshift

        self.specfitcanvas.specfit_plot(self.in_dict)
        self.specfitcanvas.setFocus()

    def remove_last_layout(self, layout):
        """Remove the layout added to the GUI's general layout.

        Parameters
        ----------
        layout : obj, layout
            The layout to be removed
        """

        # Remove mode specific layout and widgets
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget(widget_to_remove)
            # remove it from the gui
            widget_to_remove.setParent(None)
        self.layout.removeItem(layout)

    def on_press_main(self, event):
        """ Set the key functionality in the "main" mode

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        # Save the last key function and help label to return to
        # from the mask and zoom modes
        self.last_on_press = self.on_press_main
        self.last_help_label_function = self.main_help_labels

        if event.key == "q":
            # Quit
            self.close()

        elif event.key == "m":
            # Mask mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.set_mask_hbox()
            self.statusBar().showMessage("Masking mode", 5000)
            self.cid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.on_press_mask)
            self.update_help_window(self.mask_help_labels)

        elif event.key == "e":
            # Zoom
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.statusBar().showMessage("Mode: Set wavelength range to display", 5000)
            self.update_help_window(self.ranges_help_labels)
            self.handleStart(self.on_press_main)

        elif event.key == "S":
            # Save fit
            self.save_fit()

        elif event.key == "L":
            # Load fit
            self.load_fit()

        elif event.key == "c":
            # Continuum model mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.set_cont_fit_box()
            self.statusBar().showMessage("Continuum fitting mode", 5000)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.on_press_cont_fit)
            self.update_help_window(self.cont_fit_help_labels)

        elif event.key == "l":
            # Line model mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.set_line_fit_box()
            self.statusBar().showMessage("Line fitting mode", 5000)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.on_press_line_fit)
            self.update_help_window(self.line_fit_help_labels)

        elif event.key == "?":
            # Toggle help window
            self.help_window_toggle(self.main_help_labels)


    def main_help_labels(self):
        """ Set the labels for the help window

        """

        l1 = QLabel("Hot Keys - Main Mode")
        l1.setAlignment(QtCore.Qt.AlignHCenter)

        l2 = QLabel("m : Start masking mode")
        l2.setAlignment(QtCore.Qt.AlignHCenter)
        l2.setWordWrap(True)

        l3 = QLabel("e : Change displayed wavelength range")
        l3.setAlignment(QtCore.Qt.AlignHCenter)
        l3.setWordWrap(True)

        l7 = QLabel("c : Start continuum fitting")
        l7.setAlignment(QtCore.Qt.AlignHCenter)
        l7.setWordWrap(True)

        l8 = QLabel("e : Start line fitting")
        l8.setAlignment(QtCore.Qt.AlignHCenter)
        l8.setWordWrap(True)

        l4 = QLabel("S : Save fit")
        l4.setAlignment(QtCore.Qt.AlignHCenter)
        l4.setWordWrap(True)

        l5 = QLabel("L : Load fit")
        l5.setAlignment(QtCore.Qt.AlignHCenter)
        l5.setWordWrap(True)

        l6 = QLabel("q : Quit")
        l6.setAlignment(QtCore.Qt.AlignHCenter)
        l6.setWordWrap(True)

        l9 = QLabel("? : Toggle hot key info")
        l9.setAlignment(QtCore.Qt.AlignHCenter)
        l9.setWordWrap(True)

        return [l1, l2, l3, l7, l8, l4, l5, l6, l9]




    def save_fit(self):
        """ Save the current fit to a folder

        """

        # Calculate best-fit models before saving
        # to ensure the best-fit models in in_dict are populated
        # try:
        #     self.fit_cont_model()
        # except:
        #     print("Continuum model could not be fitted")
        # try:
        #     self.fit_line_model()
        # except:
        #     print("Line model cound not be fitted")

        # Select save folder
        foldername = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        # Delete all previous json files in the folder
        filelist = os.listdir(foldername)

        for file in filelist:
            if file.endswith(".json"):
                os.remove(os.path.join(foldername, file))

        # Create the hdf5 extension with the spectra, masks and fits
        disp = self.in_dict['spec'].dispersion
        flux = self.in_dict['spec'].flux
        mask_spec = self.in_dict['mask_list'][0]
        mask_cont = self.in_dict['mask_list'][1]
        mask_emline = self.in_dict['mask_list'][2]

        df = pd.DataFrame(np.array([disp,
                                   flux,
                                   mask_spec,
                                   mask_cont,
                                   mask_emline]).T,
                          columns=['dispersion',
                                   'flux',
                                   'mask_spec',
                                   'mask_cont',
                                   'mask_emline'])

        if hasattr(self.in_dict['spec'],'flux_err'):
            df['flux_err'] = self.in_dict['spec'].flux_err

        if self.in_dict['cont_init_spec'] is not None:
            cont_init_spec = self.in_dict['cont_init_spec'].flux
            df['cont_init_spec'] = cont_init_spec
        if self.in_dict['cont_fit_spec'] is not None:
            cont_fit_spec = self.in_dict['cont_fit_spec'].flux
            df['cont_fit_spec'] = cont_fit_spec
        if self.in_dict['line_init_spec'] is not None:
            line_init_spec = self.in_dict['line_init_spec'].flux
            df['line_init_spec'] = line_init_spec
        if self.in_dict['line_fit_spec'] is not None:
            line_fit_spec = self.in_dict['line_fit_spec'].flux
            df['line_fit_spec'] = line_fit_spec

        df.to_hdf(foldername+'/fit.hdf5', 'data')

        # Create the hdf5 extension with the SpecFitGui parameters
        specfit_params = {'cont_fit_z_flag': self.cont_fit_z_flag,
                          'line_fit_z_flag': self.line_fit_z_flag,
                          'fit_with_weights': self.fit_with_weights}
        df = pd.DataFrame.from_dict(specfit_params,
                                    orient='index',
                                    columns=['value'])

        df.to_hdf(foldername+'/fit.hdf5','params')

        # Save the fit results and models to json files
        if self.cont_fit_result is not None:
            save_modelresult(self.cont_fit_result, foldername+'/fit_cont_result.json')

        if self.line_fit_result is not None:
            save_modelresult(self.line_fit_result, foldername+'/fit_lines_result.json')

        if len(self.cont_model_list) > 0:
            for model in self.cont_model_list:
                model_fname = model.prefix
                save_model(model, foldername+'/cont_'+model_fname+'.json')

        if len(self.line_model_list) > 0:
            for model in self.line_model_list:
                model_fname = model.prefix
                save_model(model,foldername+'/line_'+model_fname+'.json')



    def load_fit(self):
        """ Load spectrum and fit models from a folder

        """
        # Select load folder
        foldername = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        # Read the hdf5 data extension and populate spectra, masks and fits
        df = pd.read_hdf(foldername+'/fit.hdf5', 'data')

        disp = df['dispersion'].values
        flux = df['flux'].values



        self.in_dict['spec'] = sod.SpecOneD(dispersion=disp, flux=flux, unit='f_lam')

        if hasattr(df, 'flux_err'):
            flux_err = df['flux_err'].values
            self.in_dict['spec'].flux_err = flux_err

        spec = self.in_dict['spec']

        self.in_dict['mask_list'][0] = np.array(df['mask_spec'].values,
                                                dtype=bool)
        self.in_dict['mask_list'][1] = np.array(df['mask_cont'].values,
                                                dtype=bool)
        self.in_dict['mask_list'][2] = np.array(df['mask_emline'].values,
                                                dtype=bool)

        self.in_dict['spec'].mask = self.in_dict['mask_list'][0]

        if hasattr(df, 'cont_fit_spec'):
            cont_fit_flux = df['cont_fit_spec'].values
            cont_fit_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=cont_fit_flux, unit='f_lam')
            self.in_dict['cont_fit_spec'] = cont_fit_spec

        if hasattr(df, 'cont_init_spec'):
            cont_init_flux = df['cont_init_spec'].values
            cont_init_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=cont_init_flux, unit='f_lam')
            self.in_dict['cont_init_spec'] = cont_init_spec

        if hasattr(df, 'line_fit_spec'):
            line_fit_flux = df['line_fit_spec'].values
            line_fit_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=line_fit_flux, unit='f_lam')
            self.in_dict['line_fit_spec'] = line_fit_spec

        if hasattr(df, 'line_init_spec'):
            line_init_flux = df['line_init_spec'].values
            line_init_spec = sod.SpecOneD(dispersion=spec.dispersion, flux=line_init_flux, unit='f_lam')
            self.in_dict['line_init_spec'] = line_init_spec

        # Read the hdf5 parameter extension and populate SpecFitGui parameters
        df = pd.read_hdf(foldername+'/fit.hdf5', 'params')
        self.cont_fit_z_flag = df.loc['cont_fit_z_flag', 'value']
        self.line_fit_z_flag = df.loc['line_fit_z_flag', 'value']
        self.fit_with_weights = df.loc['fit_with_weights', 'value']

        print(self.fit_with_weights)



        # Create the model list and read the json files
        # for the continuum and line models
        self.cont_model_list = []
        self.cont_model_par_list = []
        self.line_model_list = []
        self.line_model_par_list = []

        model_dict = {'power_law': power_law,
                      'power_law_continuum': power_law,
                      'power_law_at_2500A': power_law_at_2500A,
                      'template_model': template_model,
                      'template_model_new': template_model_new,
                      'gaussian_fwhm_km_s': gaussian_fwhm_km_s,
                      'gaussian_fwhm_km_s_z': gaussian_fwhm_km_s_z,
                      'gaussian_fwhm_z': gaussian_fwhm_z,
                      'balmer_continuum_model': balmer_continuum_model,
                      'power_law_at_2500A_plus_BC':
                          power_law_at_2500A_plus_BC,
                      'power_law_at_2500A_plus_flexible_BC':
                          power_law_at_2500A_plus_flexible_BC,
                      'power_law_at_2500A_plus_manual_BC':
                          power_law_at_2500A_plus_manual_BC,
                      'CIII_model_func': CIII_model_func}

        cont_models = glob.glob(foldername+'/cont_*.json')

        for model_fname in cont_models:
            model = load_model(model_fname, funcdefs=model_dict)

            # if self.cont_fit_z_flag:
            params = Parameters()
            params.add('z', value=0, min=0, max=1000, vary=True)
            pars = model.make_params()
            for p in pars:
                params.add(pars[p])
            # else:
            #     params = model.make_params()

            # params = model.make_params()
            print(params)
            self.cont_model_list.append(model)
            self.cont_model_par_list.append(params)

        line_models = glob.glob(foldername+'/line_*.json')

        for model_fname in line_models:
            model = load_model(model_fname, funcdefs=model_dict)

            # if self.line_fit_z_flag:
            params = Parameters()
            params.add('z',value=0, min=0, max=1000, vary=True)
            pars = model.make_params()
            for p in pars:
                params.add(pars[p])
            # else:
            #     params = model.make_params()

            self.line_model_list.append(model)
            self.line_model_par_list.append(params)

        # Read in the fit result files, if possible and update
        # the continuum and line model parameters
        if os.path.isfile(foldername+'/fit_cont_result.json'):
            self.cont_fit_result = load_modelresult(foldername+'/fit_cont_result.json', funcdefs=model_dict)
            self.upd_cont_param_values_from_fit()

        if os.path.isfile(foldername+'/fit_lines_result.json'):
            self.line_fit_result = load_modelresult(foldername+'/fit_lines_result.json', funcdefs=model_dict)
            self.upd_line_param_values_from_fit()

        self.redraw()


    def deleteItemsOfLayout(self, layout):
         if layout is not None:
             while layout.count():
                 item = layout.takeAt(0)
                 widget = item.widget()
                 if widget is not None:
                     widget.setParent(None)
                 else:
                     self.deleteItemsOfLayout(item.layout())

# ------------------------------------------------------------------------------
#  HELP WINDOW FUNCTIONS
# ------------------------------------------------------------------------------


    def help_window_toggle(self, label_function):
        """ Toggle the help window on or off

        Parameters
        ----------
        label_function : function
            Function, that creates the content of the help window
            according to the currently active mode.
        """
        if self.help_window_flag == False:
            self.help_window_flag = True
            self.open_help_window(label_function)

        elif self.help_window_flag == True:
            self.help_window_flag = False
            self.close_help_window()


    def open_help_window(self, label_function):
        """ Open the help window

         Parameters
         ----------
         label_function : function
             Function, that creates the content of the help window
             according to the currently active mode.
         """

        self.help_box = QVBoxLayout()
        label_list = label_function()

        for label in label_list:
            self.help_box.addWidget(label)

        self.plot_box.addLayout(self.help_box)

    def close_help_window(self):
        """ Close the help window

        """

        self.remove_last_layout(self.help_box)


    def update_help_window(self, label_function):
        """ Update the contents of the help window

        """

        if self.help_window_flag == True:
            self.close_help_window()
            self.open_help_window(label_function)

# ------------------------------------------------------------------------------
# DISPERSION ZOOM FUNCTIONS
# ------------------------------------------------------------------------------


    def handleStart(self, on_press):
        """ Start an interactive handling loop, which closes once a lower
        and an upper dispersion boundary have been interactively defined. It
        also closes on pressing "r", resetting the plotting boundaries.

        """

        self._wx1 = None
        self._wx2 = None
        self._running = True
        self.cid = self.specfitcanvas.mpl_connect('key_press_event', self.on_press_set_ranges)
        while self._running:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)

            if (self._wx1 is not None and self._wx2 is not None):
                self.in_dict['x_lo'] = self._wx1
                self.in_dict['x_hi'] = self._wx2
                self._running = False
                self.statusBar().showMessage("Zommed to %d - %d" %
                                             (self._wx1, self._wx2), 5000)
                self.specfitcanvas.mpl_disconnect(self.cid)
                self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                           on_press)
                self.update_help_window(self.last_help_label_function)
                self.redraw()

    def handleStop(self):
        """ Stop the interactive loop.

        """
        self._running = False

    def on_press_set_ranges(self, event):
        """ Set the key functionality in the zoom dispersion mode

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key == "A":
            # Set lower dispersion limit
            self._wx1 = event.xdata
            self.statusBar().showMessage("Setting first limit to %d" %
                                         (event.xdata), 2000)

        elif event.key == "D":
            # Set upper dispersion limit
            self._wx2 = event.xdata
            self.statusBar().showMessage("Setting second limit to %d" %
                                         (event.xdata), 2000)

        elif event.key == "r":
            # Reset dispersion limits
            self.statusBar().showMessage("Resetting wavelength range "
                                         "display limits", 2000)

            self.in_dict['x_lo'] = min(self.in_dict['spec'].dispersion)
            self.in_dict['x_hi'] = max(self.in_dict['spec'].dispersion)
            self.in_dict['y_lo'] = min(self.in_dict['spec'].flux)
            self.in_dict['y_hi'] = max(self.in_dict['spec'].flux)

            self.redraw()
            self.handleStop()

            self.specfitcanvas.mpl_disconnect(self.cid)
            self.statusBar().showMessage(
                "Reset wavelength range limits", 5000)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.last_on_press)
            self.update_help_window(self.last_help_label_function)
            self.redraw()

        elif event.key == "q":
            # Quit dispersion zoom mode
            self.handleStop()

            self.specfitcanvas.mpl_disconnect(self.cid)
            self.statusBar().showMessage("Stopped setting wavelength range limits", 5000)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event', self.last_on_press)
            self.update_help_window(self.last_help_label_function)
            self.redraw()

        elif event.key == "?":
            # Toggle help window
            self.help_window_toggle(self.ranges_help_labels)

    def ranges_help_labels(self):
        """ Set the labels for the help window

        """

        l1 = QLabel("Hot Keys")
        # l1.setAlignment(QtCore.Qt.AlignTop)
        l1.setAlignment(QtCore.Qt.AlignHCenter)

        l1b = QLabel("Set Wavelength Range Mode")
        # l1b.setAlignment(QtCore.Qt.AlignTop)
        l1b.setAlignment(QtCore.Qt.AlignHCenter)

        l2 = QLabel("A : Select first wavelength range limit")
        l2.setAlignment(QtCore.Qt.AlignHCenter)
        l2.setWordWrap(True)

        l3 = QLabel("D : Select second wavelength range limit")
        l3.setAlignment(QtCore.Qt.AlignHCenter)
        l3.setWordWrap(True)

        l4 = QLabel("r : Reset wavelength range")
        l4.setAlignment(QtCore.Qt.AlignHCenter)
        l4.setWordWrap(True)

        l5 = QLabel("q : Quit to main mode")
        l5.setAlignment(QtCore.Qt.AlignHCenter)
        l5.setWordWrap(True)

        l6 = QLabel("? : Toggle hot key info")
        l6.setAlignment(QtCore.Qt.AlignHCenter)
        l6.setWordWrap(True)

        return [l1, l1b, l2, l3, l4, l5, l6]


# ------------------------------------------------------------------------------
# MASK MODE FUNCTIONS
# ------------------------------------------------------------------------------

    def set_mask_hbox(self):
        """ Set up the hbox layout for the masking mode of the GUI.

        """

        # Adding masking values to the GUI
        self.mask_lo_lbl = QLabel("Mask, lower edge:")
        self.mask_lo_val = QLabel("{0:.2f}".format(self.mx1))
        self.mask_hi_lbl = QLabel("Mask, higher edge:")
        self.mask_hi_val = QLabel("{0:.2f}".format(self.mx2))

        self.mask_lo_in = QLineEdit("{0:.2f}".format(self.mx1))
        self.mask_lo_in.setMaxLength(7)

        self.mask_hi_in = QLineEdit("{0:.2f}".format(self.mx2))
        self.mask_hi_in.setMaxLength(7)

        # Pre determined mask windows
        self.mask_combobox = QComboBox()
        self.mask_combobox.addItem("X-SHOOTER NIR atmospheric windows")
        # self.mask_combobox.addItem("QSO High-z PCA model 1200-3100")
        self.mask_combobox.addItem("X-SHOOTER HighZ continuum windows")
        self.mask_combobox.addItem("X-SHOOTER HighZ line windows")
        self.mask_combobox.addItem("QSO continuum windows (VP06)")
        self.mask_combobox.addItem("QSO continuum windows")
        self.mask_combobox.addItem("QSO MgII cont/fe windows (Shen11)")
        self.mask_combobox.addItem("QSO Hbeta cont/fe windows (Shen11)")
        self.mask_combobox.addItem("QSO Halpha cont/fe windows (Shen11)")
        self.mask_combobox.addItem("QSO CIV cont/fe windows (Shen11)")
        # self.mask_combobox.addItem("Pisco Continuum")

        self.mask_preset_apply_button = QPushButton("Apply mask preset")
        self.mask_preset_apply_button.clicked.connect(self.apply_mask_presets)


        self.maskbox = QHBoxLayout()

        for w in [self.mask_lo_lbl,
                  self.mask_lo_val,
                  self.mask_lo_in,
                  self.mask_hi_lbl,
                  self.mask_hi_val,
                  self.mask_hi_in,
                  self.mask_combobox,
                  self.mask_preset_apply_button]:
            self.maskbox.addWidget(w)
            self.maskbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.maskbox)

        self.mask_lo_in.returnPressed.connect(self.upd_mask_values)
        self.mask_hi_in.returnPressed.connect(self.upd_mask_values)


    def upd_mask_values(self):
        """ Update the masking limits to the user defined
        input values.

        """
        self.mx1 = float(self.mask_lo_in.text())
        self.mx2 = float(self.mask_hi_in.text())
        print(self.mx1, self.mx2)

        disp_min = (self.in_dict['spec'].dispersion[0])
        disp_max = (self.in_dict['spec'].dispersion[-1])
        if self.mx1 < disp_min or self.mx1 is None:
            self.mx1 = disp_min
        elif self.mx1 > disp_max:
            self.mx1 = disp_max
        if self.mx2 < disp_min:
            self.mx2 = disp_min
        elif self.mx2 > disp_max or self.mx2 is None:
            self.mx2 = disp_max

        self.mask_lo_val.setText("{0:.2f}".format(self.mx1))
        self.mask_hi_val.setText("{0:.2f}".format(self.mx2))

        # self.set_canvas_active()


    def on_press_mask(self, event):
        """ Set the key functionality in the "mask" mode

            Parameters
            ----------
            event : key_press_event
                Key press event to evaluate
        """

        if event.key == "1":
            # Set general mask active
            self.active_mask = 0
            self.statusBar().showMessage("Masking Mode: Mask out regions"
                                         " of the spectrum ", 5000)
        elif event.key == "2":
            # Set continuum model fit mask active
            self.active_mask = 1
            self.statusBar().showMessage("Masking Mode: Mask in regions"
                                         " for the continuum fit ", 5000)
        elif event.key == "3":
            # Set line model fit mask active
            self.active_mask = 2
            self.statusBar().showMessage("Masking Mode: Mask in regions"
                                         " for the line fit ", 5000)
        elif event.key == "A":
            # Set lower mask limit
            if event.xdata is not None:
                self.mx1 = event.xdata
            else:
                self.mx1 = (self.in_dict['spec'].dispersion[0])
            self.mask_lo_val.setText("{0:.2f}".format(self.mx1))
            self.mask_lo_in.setText("{0:.2f}".format(self.mx1))
            self.statusBar().showMessage(
                "Setting lower mask edge to %d" % (event.xdata), 2000)

        elif event.key == "D":
            # Set upper mask limit
            if event.xdata is not None:
                self.mx2 = event.xdata
            else:
                self.mx2 = (self.in_dict['spec'].dispersion[-1])
            self.mask_hi_val.setText("{0:.2f}".format(self.mx2))
            self.mask_hi_in.setText("{0:.2f}".format(self.mx2))
            self.statusBar().showMessage("Setting higher mass edge to %d" % (event.xdata), 2000)

        elif event.key == "m":
            # Mask in/out specified region in active mask
            self.mask()

        elif event.key == "u":
            # Mask out/in specified region in active mask
            self.unmask()

        elif event.key == "r":
            # Reset active mask
            if self.active_mask == 0:
                self.in_dict['mask_list'][self.active_mask] = np.ones(self.in_dict['spec'].dispersion.shape, dtype=bool)
            else :
                self.in_dict['mask_list'][self.active_mask] = np.zeros(self.in_dict['spec'].dispersion.shape, dtype=bool)
            self.in_dict['spec'].mask = self.in_dict['mask_list'][0]
            self.redraw()

        elif event.key == "R":
            # Reset all masks
            for idx, mask in enumerate(self.in_dict['mask_list']):
                if idx == 0:
                    self.in_dict['mask_list'][idx] = np.ones(self.in_dict['spec'].dispersion.shape, dtype=bool)
                else :
                    self.in_dict['mask_list'][idx] = np.zeros(self.in_dict['spec'].dispersion.shape, dtype=bool)
            self.redraw()

        elif event.key == "q":
            # Quit mask mode
            self.specfitcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.maskbox)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event', self.last_on_press)
            self.update_help_window(self.last_help_label_function)
            self.statusBar().showMessage("SpecFitGui Main Mode", 5000)

        elif event.key == "?":
            # Toggle help window
            self.help_window_toggle(self.mask_help_labels)

    def mask_help_labels(self):
        """ Set the labels for the help window

        """

        l1 = QLabel("Hot Keys - Mask Mode")
        l1.setAlignment(QtCore.Qt.AlignHCenter)

        l2 = QLabel("1 : Select general mask for spectrum ")
        l2.setAlignment(QtCore.Qt.AlignHCenter)
        l2.setWordWrap(True)

        l3 = QLabel("2 : Select region mask for continuum fit")
        l3.setAlignment(QtCore.Qt.AlignHCenter)
        l3.setWordWrap(True)

        l4 = QLabel("3 : Select region mask for line fit")
        l4.setAlignment(QtCore.Qt.AlignHCenter)
        l4.setWordWrap(True)

        l5 = QLabel("A : Select lower mask boundary")
        l5.setAlignment(QtCore.Qt.AlignHCenter)
        l5.setWordWrap(True)

        l6 = QLabel("D : Select upper mask boundary")
        l6.setAlignment(QtCore.Qt.AlignHCenter)
        l6.setWordWrap(True)

        l7 = QLabel("m : Apply selection to current mask")
        l7.setAlignment(QtCore.Qt.AlignHCenter)
        l7.setWordWrap(True)

        l8 = QLabel("u : Unmask selection in current mask")
        l8.setAlignment(QtCore.Qt.AlignHCenter)
        l8.setWordWrap(True)

        l9 = QLabel("r : Reset current mask")
        l9.setAlignment(QtCore.Qt.AlignHCenter)
        l9.setWordWrap(True)

        l10 = QLabel("R : Reset all masks")
        l10.setAlignment(QtCore.Qt.AlignHCenter)
        l10.setWordWrap(True)

        l11 = QLabel("q : Quit")
        l11.setAlignment(QtCore.Qt.AlignHCenter)
        l11.setWordWrap(True)

        l12 = QLabel("? : Toggle hot key info")
        l12.setAlignment(QtCore.Qt.AlignHCenter)
        l12.setWordWrap(True)

        return [l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12]

    def apply_mask_presets(self):
        """ Add the mask preset to the specified region in the active spectrum

        :return:
        """

        # QSO continuum windows, see Vestergaard & Peterson 2006
        qso_cont_VP06 = np.array([[1265, 1290], [1340, 1375], [1425, 1470],
                                [1680, 1705], [1905, 2050]])
        qso_contfe_mgII_Shen11 = np.array([[2200, 2700], [2900, 3090]])

        qso_contfe_hbeta_Shen11 = np.array([[4435, 4700], [5100, 5535]])

        qso_contfe_halpha_Shen11 = np.array([[6000, 6250], [6800, 7000]])

        qso_contfe_cIV_Shen11 = np.array([[1445, 1465], [1700, 1705]])

        xshooter_nir_atmospheric_windows = np.array([[5000, 10250],
                                                     [13450, 14300],
                                                     [18000, 19400],
                                                     [22600, 30000]])

        xshooter_surge_continuum_windows = np.array([[1445, 1465],
                                                     [1700, 1705],
                                                     [2155, 2400],
                                                     [2480, 2675],
                                                     [2925, 3100]])

        xshooter_surge_line_windows = np.array([[1340, 1425],
                                                [1470, 1600],
                                                [1800, 2000],
                                                [2700, 2900]])

        qso_continuum_windows = np.array([
                                          [1350, 1360],
                                          [1445, 1465],
                                          [1700, 1705],
                                          [2155, 2400],
                                          [2480, 2675],
                                          [2925, 3100]])

        # xshooter_surge_line_windows = np.array([[2700, 2900],
        #                                         [1470, 1600]])

        # qso_pca_1200_3100_model = np.array([[0, 1200],
        #                                     [3100, 99999]])

        pisco_continuum = np.array([[1275, 1285], [1310, 1325], [2500,2750],
                                    [2850, 2890]])

        mask_preset_name = self.mask_combobox.currentText()
        redsh = self.redshift

        previous_active_mask = self.active_mask

        if mask_preset_name == "QSO continuum windows (VP06)":
            masks_to_apply = (1. + redsh) * qso_cont_VP06
            self.active_mask = 1
        elif mask_preset_name == "QSO MgII cont/fe windows (Shen11)":
            masks_to_apply = (1. + redsh) * qso_contfe_mgII_Shen11
            self.active_mask = 1
        elif mask_preset_name == "QSO Hbeta cont/fe windows (Shen11)":
            masks_to_apply = (1. + redsh) * qso_contfe_hbeta_Shen11
            self.active_mask = 1
        elif mask_preset_name == "QSO Halpha cont/fe windows (Shen11)":
            masks_to_apply = (1. + redsh) * qso_contfe_halpha_Shen11
            self.active_mask = 1
        elif mask_preset_name == "QSO CIV cont/fe windows (Shen11)":
            masks_to_apply = (1. + redsh) * qso_contfe_cIV_Shen11
            self.active_mask = 1
        elif mask_preset_name == "X-SHOOTER NIR atmospheric windows":
            masks_to_apply = xshooter_nir_atmospheric_windows
            self.active_mask = 0
        # elif mask_preset_name == "QSO High-z PCA model 1200-3100":
        #     masks_to_apply = (1. + redsh) * qso_pca_1200_3100_model
        #     self.active_mask = 0
        elif mask_preset_name == "X-SHOOTER HighZ continuum windows":
            masks_to_apply = (1. + redsh) * xshooter_surge_continuum_windows
            self.active_mask = 1
        elif mask_preset_name == "QSO continuum windows":
            masks_to_apply = (1. + redsh) * qso_continuum_windows
            self.active_mask = 1
        elif mask_preset_name == "X-SHOOTER HighZ line windows":
            masks_to_apply = (1. + redsh) * xshooter_surge_line_windows
            self.active_mask = 2
        elif mask_preset_name == "Pisco Continuum":
            masks_to_apply = (1. + redsh) * pisco_continuum
            self.active_mask = 1



        for mask in masks_to_apply:
            self.mx1 = mask[0]
            self.mx2 = mask[1]

            self.mask_without_redraw()

        self.active_mask = previous_active_mask
        self.redraw()


    def mask_without_redraw(self):

        self.statusBar().showMessage(
            "Masking spectral region between %d and %d" % (self.mx1, self.mx2),
            2000)

        mask_between = np.sort(np.array([self.mx1, self.mx2]))
        spec = self.in_dict['spec']
        lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))

        if self.active_mask == 0:
            self.in_dict['mask_list'][self.active_mask][
            lo_index:up_index] = False
        else:
            self.in_dict['mask_list'][self.active_mask][
            lo_index:up_index] = True

        self.in_dict['spec'].mask = self.in_dict['mask_list'][0]

    def mask(self):

        """ Mask in/out the specified region in the active spectrum.

        """
        self.mask_without_redraw()

        self.redraw()

    def unmask(self):
        """ Mask out/in the specified region in the active spectrum.

        """
        self.statusBar().showMessage("Masking spectral region between %d and %d" % (self.mx1, self.mx2), 2000)

        mask_between = np.sort(np.array([self.mx1, self.mx2]))
        spec = self.in_dict['spec']
        lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))

        if self.active_mask == 0:
            self.in_dict['mask_list'][self.active_mask][lo_index:up_index] = True
        else :
            self.in_dict['mask_list'][self.active_mask][lo_index:up_index] = False

        self.in_dict['spec'].mask = self.in_dict['mask_list'][0]
        self.redraw()

# ------------------------------------------------------------------------------
# CONTINUUM MODEL FIT FUNCTIONS
# ------------------------------------------------------------------------------

    def set_cont_fit_box(self):
        """Create the vertical gui box for the continuum fitting mode

        """

        self.cont_fit_vbox = QVBoxLayout()
        self.cont_fit_vbox_lines = QHBoxLayout()


        self.cont_fit_box = QComboBox()
        self.cont_fit_box.addItem("Power Law")
        self.cont_fit_box.addItem("Power Law (2500A)")
        self.cont_fit_box.addItem("Power Law (2500A + BC 30%)")
        self.cont_fit_box.addItem("Power Law (2500A + flexible BC)")
        self.cont_fit_box.addItem("Power Law (2500A + manual BC)")
        self.cont_fit_box.addItem("Balmer Continuum")
        self.cont_fit_box.addItem("Tsuzuki06")
        self.cont_fit_box.addItem("Vestergaard 01")
        self.cont_fit_box.addItem("Simqso Fe template")
        self.cont_fit_box.addItem("Full subdivided iron template")
        self.cont_fit_box.addItem("Iron template 1200-2200")
        self.cont_fit_box.addItem("Iron template 1200-2200 (cont.)")
        self.cont_fit_box.addItem("Iron template 2200-3500 (T06)")
        self.cont_fit_box.addItem("Iron template 2200-3500 (V01)")
        self.cont_fit_box.addItem("Iron template 2200-3500 (T06, cont.)")
        self.cont_fit_box.addItem("Iron template 2200-3500 (V01, cont.)")
        self.cont_fit_box.addItem("Iron template 2200-3500 new (T06, cont.)")
        self.cont_fit_box.addItem("Iron template 2200-3500 new (V01, cont.)")
        self.cont_fit_box.addItem("Iron template 3700-5600")
        self.cont_fit_box.addItem("Iron template 3700-5600 (cont.)")
        self.cont_fit_box.addItem("Iron template 3700-5600 new (cont.)")
        # self.cont_fit_box.addItem("FeIII VW01")
        # self.cont_fit_box.addItem("FeII BG92")


        self.cont_prefix_label = QLabel("Prefix")
        self.cont_prefix_in = QLineEdit('{}'.format("c"))
        self.cont_prefix_in.setMaxLength(12)

        self.cont_add_button = QPushButton("+")
        self.cont_add_button.clicked.connect(self.add_cont_to_model)
        self.cont_remove_button = QPushButton("-")
        self.cont_remove_button.clicked.connect(self.remove_last_cont_from_model)
        self.cont_del_model_box = QComboBox()
        self.cont_del_model_button = QPushButton("Delete model")
        self.cont_del_model_button.clicked.connect(self.delete_selected_cont_model)

        self.fluxerrweights_cont_checkbox = QCheckBox('Fit with weights (flux '
                                                'errors)')
        self.fluxerrweights_cont_checkbox.setChecked(self.fit_with_weights)
        self.fluxerrweights_cont_checkbox.stateChanged.connect(
            self.upd_cont_values)

        for w in [self.cont_fit_box, self.cont_prefix_label,
                  self.cont_prefix_in, self.cont_add_button,
                  self.cont_remove_button, self.cont_del_model_box,
                  self.cont_del_model_button,
                  self.fluxerrweights_cont_checkbox]:
            self.cont_fit_vbox.addWidget(w)

        self.cont_fit_hbox = QHBoxLayout()
        self.cont_fit_hbox.addLayout(self.cont_fit_vbox,1)
        self.cont_fit_hbox.addLayout(self.cont_fit_vbox_lines,2)

        self.layout.addLayout(self.cont_fit_hbox)

        self.cont_fit_z_lineditlist = []
        self.cont_fit_z_varyboxlist = []

        self.set_hbox_cont_fit()
        self.add_cont_hbox_redshift()

        # If continuum models and parameters already exist
        # populate the continuum mode with those models/parameters
        if self.cont_model_par_list is not None and \
                self.cont_model_list is not None:
            self.set_hbox_cont_fit()
            self.restore_hbox_cont_fit()



    def add_cont_hbox_redshift(self):
        """Create the gui box for the redshift parameter

         """

        label = QLabel('z')
        linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].value))
        linedit.setMaxLength(20)
        min_label = QLabel("min")
        min_linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].min))
        min_linedit.setMaxLength(20)
        max_label = QLabel("max")
        max_linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].max))
        max_linedit.setMaxLength(20)
        vary_checkbox = QCheckBox("vary")
        vary_checkbox.setChecked(self.redsh_par['z'].vary)
        fit_z_checkbox = QCheckBox("fit z (overrides model z parameters)")
        fit_z_checkbox.setChecked(self.cont_fit_z_flag)

        widgetlist = [label, linedit, min_label, min_linedit, max_label,
                      max_linedit, vary_checkbox, fit_z_checkbox]

        self.cont_fit_z_lineditlist = [linedit, min_linedit, max_linedit]
        self.cont_fit_z_varyboxlist = [vary_checkbox, fit_z_checkbox]

        for l in [linedit, min_linedit, max_linedit]:
            l.returnPressed.connect(self.upd_cont_values)

        vary_checkbox.stateChanged.connect(self.upd_cont_values)
        fit_z_checkbox.stateChanged.connect(self.upd_cont_values)


        hbox_redshift = QVBoxLayout()
        for w in widgetlist:
            hbox_redshift.addWidget(w)
            hbox_redshift.setAlignment(w,  QtCore.Qt.AlignVCenter)


        self.cont_fit_vbox.addLayout(hbox_redshift)

    def set_hbox_cont_fit(self):
        """Create the empty widget lists for the continuum mode

        """
        self.cont_fit_hbox_widgetlist = []
        self.cont_fit_lineditlist = []
        self.cont_fit_varybox_list = []
        self.cont_fit_hbox_list = []

    def upd_cont_del_box(self):
        """ Update the delete drop down box with current continuum models

        """

        self.cont_del_model_box.clear()
        for idx, model in enumerate(self.cont_model_list):
            self.cont_del_model_box.addItem(model.prefix)

    def restore_hbox_cont_fit(self):
        """ Restore the continuum fit hbox from the main menu or after loading
        a previous fit.

        """
        self.upd_cont_del_box()

        for idx, params in enumerate(self.cont_model_par_list):

            widgetlist = []
            lineditlist = []
            varyboxlist = []

            for param in params :
                label = QLabel(param)
                linedit = QLineEdit('{:.4E}'.format(params[param].value))
                linedit.setMaxLength(20)
                expr_linedit = QLineEdit('{}'.format(params[param].expr))
                expr_linedit.setMaxLength(20)
                min_label = QLabel("min")
                min_linedit = QLineEdit('{:.4E}'.format(params[param].min))
                min_linedit.setMaxLength(20)
                max_label = QLabel("max")
                max_linedit = QLineEdit('{:.4E}'.format(params[param].max))
                max_linedit.setMaxLength(20)
                vary_checkbox = QCheckBox("vary")
                vary_checkbox.setChecked(params[param].vary)

                widgetlist.extend([label, linedit, expr_linedit, min_label, min_linedit, max_label, max_linedit, vary_checkbox])
                lineditlist.extend([linedit, expr_linedit, min_linedit, max_linedit])
                varyboxlist.append(vary_checkbox)

                vary_checkbox.stateChanged.connect(self.upd_cont_values)

            self.cont_fit_hbox_widgetlist.extend(widgetlist)
            self.cont_fit_lineditlist.append(lineditlist)
            self.cont_fit_varybox_list.append(varyboxlist)

            cont_groupbox = QGroupBox()
            layout_groupbox = QVBoxLayout(cont_groupbox)
            vbox = QVBoxLayout()

            for w in widgetlist:
                layout_groupbox.addWidget(w)
                layout_groupbox.setAlignment(w,  QtCore.Qt.AlignVCenter)

            ScArea = QScrollArea()
            ScArea.setLayout(QVBoxLayout())
            ScArea.setWidgetResizable(True)
            ScArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

            ScArea.setWidget(cont_groupbox)

            vbox.addWidget(ScArea)
            self.cont_fit_hbox_list.append(vbox)

            self.cont_fit_vbox_lines.addLayout(vbox)

            for l in lineditlist:
                l.returnPressed.connect(self.upd_cont_values)

    def on_press_cont_fit(self, event):
        """ Set the key functionality in the "continuum model" mode

            Parameters
            ----------
            event : key_press_event
                Key press event to evaluate
        """

        self.last_on_press = self.on_press_cont_fit
        self.last_help_label_function = self.cont_fit_help_labels

        if event.key == "q":
            # Quit continuum fit mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.deleteItemsOfLayout(self.cont_fit_hbox)
            self.remove_last_layout(self.cont_fit_hbox)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event', self.on_press_main)
            self.statusBar().showMessage("SpecFitGui Main Mode", 5000)
            self.last_on_press = self.on_press_main
            self.update_help_window(self.main_help_labels)

        elif event.key == "e":
            # Change to dispersion zoom mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.statusBar().showMessage("Mode: Set wavelength range to display", 5000)
            self.update_help_window(self.ranges_help_labels)
            self.handleStart(self.on_press_cont_fit)

        elif event.key == "m":
            # Change to masking mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.set_mask_hbox()
            self.statusBar().showMessage("Masking mode", 5000)
            self.cid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.on_press_mask)
            self.update_help_window(self.mask_help_labels)

        elif event.key == "f":
            # Fit continuum model
            self.fit_cont_model()

        elif event.key == "S":
            # Save fit to folder
            self.save_fit()

        elif event.key == "?":
            # Toggle help window
            self.help_window_toggle(self.cont_fit_help_labels)

    def cont_fit_help_labels(self):
        """ Set the labels for the help window

        """

        l1 = QLabel("Hot Keys - Continuum Fitting")
        l1.setAlignment(QtCore.Qt.AlignHCenter)

        l2 = QLabel("e : Change displayed wavelength range")
        l2.setAlignment(QtCore.Qt.AlignHCenter)
        l2.setWordWrap(True)

        l3 = QLabel("m : Start masking mode")
        l3.setAlignment(QtCore.Qt.AlignHCenter)
        l3.setWordWrap(True)

        l4 = QLabel("f : Fit continuum model")
        l4.setAlignment(QtCore.Qt.AlignHCenter)
        l4.setWordWrap(True)

        l5 = QLabel("S : Save fit")
        l5.setAlignment(QtCore.Qt.AlignHCenter)
        l5.setWordWrap(True)

        l6 = QLabel("q : Quit")
        l6.setAlignment(QtCore.Qt.AlignHCenter)
        l6.setWordWrap(True)

        l7 = QLabel("? : Toggle hot key info")
        l7.setAlignment(QtCore.Qt.AlignHCenter)
        l7.setWordWrap(True)

        return [l1, l2, l3, l4, l5, l6, l7]

    def upd_cont_values(self):
        """ Update the parameter values for all continuum models from the
        line input of the GUI.

        """

        # Update fit with errors
        self.fit_with_weights = self.fluxerrweights_cont_checkbox.isChecked()
        print("UPD CONT VALUES", self.fit_with_weights)
        # Update redshift parameter + fit flag
        new_z = float(self.cont_fit_z_lineditlist[0].text())
        new_z_min = float(self.cont_fit_z_lineditlist[1].text())
        new_z_max =float(self.cont_fit_z_lineditlist[2].text())
        new_z_vary = self.cont_fit_z_varyboxlist[0].isChecked()
        new_z_fit = self.cont_fit_z_varyboxlist[1].isChecked()

        self.redsh_par['z'].set(value=new_z, min=new_z_min, max=new_z_max, vary=new_z_vary )

        self.cont_fit_z_flag = new_z_fit

        # Update all parameters in cont_model_par_list
        for idx, cont_model_pars in enumerate(self.cont_model_par_list):
            params = self.cont_model_par_list[idx]

            for jdx, param in enumerate(params):
                try:
                    new_value = float(self.cont_fit_lineditlist[idx][jdx*4+0].text())
                    new_expr = self.cont_fit_lineditlist[idx][jdx*4+1].text()
                    new_min = float(self.cont_fit_lineditlist[idx][jdx*4+2].text())
                    new_max = float(self.cont_fit_lineditlist[idx][jdx*4+3].text())
                    new_vary = self.cont_fit_varybox_list[idx][jdx].isChecked()

                    if new_expr == 'None':
                        new_expr = None
                    # Set the new parameter values
                    self.cont_model_par_list[idx][param].set(value=new_value, expr=new_expr, min=new_min, max=new_max, vary=new_vary)
                except:
                    print("Input does not conform to string or float limitations!")

        self.build_full_cont_model()

    def upd_cont_param_values_from_fit(self):
        """Update the continuum model parameters from the latest best-fit
        continuum model.

        """

        # Update the redshift parameter
        if self.cont_fit_z_flag:
            temp_val = self.cont_fit_result.params['z'].value
            self.redsh_par['z'].value = temp_val


        # Update all parameters in cont_model_par_list
        for idx, cont_model in enumerate(self.cont_model_list):

            params = self.cont_model_par_list[idx]

            for jdx, param in enumerate(params):
                print(param)

                temp_val = self.cont_fit_result.params[param].value
                self.cont_model_par_list[idx][param].value = temp_val
                # print (temp_val)

                temp_val = self.cont_fit_result.params[param].expr
                self.cont_model_par_list[idx][param].expr = temp_val
                # print(temp_val)

                temp_val = self.cont_fit_result.params[param].min
                self.cont_model_par_list[idx][param].min = temp_val
                # print(temp_val)

                temp_val = self.cont_fit_result.params[param].max
                self.cont_model_par_list[idx][param].max = temp_val
                # print(temp_val)

                temp_val = self.cont_fit_result.params[param].vary
                self.cont_model_par_list[idx][param].vary = temp_val
                # print(temp_val)

    def upd_linedit_values_from_cont_params(self):
        """ Update the line edit values based on the current continuum model
        parameters.

        """

        # Update the redshift parameter
        print('fit params', self.cont_fit_result.params)

        if self.cont_fit_z_flag:
            temp_val = self.redsh_par['z'].value
            self.cont_fit_z_lineditlist[0].setText("{0:.4f}".format(temp_val))

        # Update all parameters in cont_model_par_list
        for idx, cont_model in enumerate(self.cont_model_list):

            params = self.cont_model_par_list[idx]

            for jdx, param in enumerate(params):

                temp_val = self.cont_model_par_list[idx][param].value
                self.cont_fit_lineditlist[idx][jdx*4+0].setText("{"
                                                                "0:4E}".format(temp_val))
                temp_val = self.cont_model_par_list[idx][param].expr
                self.cont_fit_lineditlist[idx][jdx*4+1].setText("{}".format(temp_val))

    def add_cont_to_model(self):
        """ Add a model to the continuum model.

         """

        cont_model_name = self.cont_fit_box.currentText()
        prefix = self.cont_prefix_in.text()+'_'
        prefix_flag = True

        # Check if chosen prefix already exists. If so, change the prefix_flag
        for model in self.cont_model_list:
            if prefix == model.prefix:
                self.statusBar().showMessage("Continuum model with same prefix"
                                             " exists already! Please choose a"
                                             " different prefix.", 5000)
                prefix_flag = False

        # If the pre_fix already exists abort, otherwise add model to
        # continuum model list including parameters.
        if prefix_flag:

            if cont_model_name == "Power Law":
                cont_params = Parameters()

                if hasattr(self, 'flux_2500'):
                    amp_guess = self.flux_2500 / \
                                (2500 * (self.redshift+1.))**(-1.5)
                else:
                    amp_guess = 2.5e-10
                cont_params.add(prefix+'amp', value=amp_guess,
                                min=amp_guess/100.,
                                max=amp_guess*100)
                cont_params.add(prefix+'slope', value=-1.5, min=-2.5,
                                max=-0.3)

                cont_model = Model(power_law, prefix=prefix)

            elif cont_model_name == "Power Law (2500A)":
                cont_params = Parameters()
                if hasattr(self, 'flux_2500'):
                    amp_guess = self.flux_2500
                else:
                    amp_guess = 2.5e-10
                if hasattr(self, 'redshift'):
                    redsh_guess = self.redshift
                else:
                    redsh_guess = 3.0

                cont_params.add('z', value=redsh_guess, min=0, max=1080,
                                vary=False)

                cont_params.add(prefix+'z', value=redsh_guess, min=0,
                                max=1080, vary=False, expr='z')

                cont_params.add(prefix+'amp', value=amp_guess,
                                )
                cont_params.add(prefix+'slope', value=-1.5, min=-2.5,
                                max=-0.3)



                cont_model = Model(power_law_at_2500A, prefix=prefix)

            elif cont_model_name == "Power Law (2500A + BC 30%)":

                cont_params = Parameters()
                if hasattr(self, 'flux_2500'):
                    amp_guess = self.flux_2500
                else:
                    amp_guess = 2.5e-10
                if hasattr(self, 'redshift'):
                    redsh_guess = self.redshift
                else:
                    redsh_guess = 3.0

                cont_params.add('z', value=redsh_guess, min=0, max=1080,
                                vary=False)

                cont_params.add(prefix + 'z', value=redsh_guess, min=0,
                                max=1080, vary=False, expr='z')

                cont_params.add(prefix + 'amp', value=amp_guess,
                                )
                cont_params.add(prefix + 'slope', value=-1.5, min=-2.5,
                                max=-0.3)

                cont_params.add(prefix + 'T_e', value=15000, min=10000,
                                max=20000, vary=False)
                cont_params.add(prefix + 'tau_BE', value=1.0, min=0.1, max=2.0,
                                vary=False)

                cont_params.add(prefix + 'lambda_BE', value=3646, vary=False)

                cont_model = Model(power_law_at_2500A_plus_BC, prefix=prefix)


            elif cont_model_name == "Power Law (2500A + flexible BC)":

                cont_params = Parameters()
                if hasattr(self, 'flux_2500'):
                    amp_guess = self.flux_2500
                else:
                    amp_guess = 2.5e-10
                if hasattr(self, 'redshift'):
                    redsh_guess = self.redshift
                else:
                    redsh_guess = 3.0

                cont_params.add('z', value=redsh_guess, min=0, max=1080,
                                vary=False)

                cont_params.add(prefix + 'z', value=redsh_guess, min=0,
                                max=1080, vary=False, expr='z')

                cont_params.add(prefix + 'amp', value=amp_guess,
                                )
                cont_params.add(prefix + 'slope', value=-1.5, min=-2.5,
                                max=-0.3)

                cont_params.add(prefix + 'f', value=0.1, min=0.0,
                                max=0.4, vary=True)

                cont_params.add(prefix + 'T_e', value=15000, min=10000,
                                max=20000, vary=False)
                cont_params.add(prefix + 'tau_BE', value=1.0, min=0.1, max=2.0,
                                vary=False)

                cont_params.add(prefix + 'lambda_BE', value=3646, vary=False)

                cont_model = Model(power_law_at_2500A_plus_flexible_BC, prefix=prefix)

            elif cont_model_name == "Power Law (2500A + manual BC)":

                cont_params = Parameters()
                if hasattr(self, 'flux_2500'):
                    amp_guess = self.flux_2500
                else:
                    amp_guess = 2.5e-10
                if hasattr(self, 'redshift'):
                    redsh_guess = self.redshift
                else:
                    redsh_guess = 3.0

                cont_params.add('z', value=redsh_guess, min=0, max=1080,
                                vary=False)

                cont_params.add(prefix + 'z', value=redsh_guess, min=0,
                                max=1080, vary=False, expr='z')

                cont_params.add(prefix + 'amp', value=amp_guess,
                                )
                cont_params.add(prefix + 'slope', value=-1.5, min=-2.5,
                                max=-0.3)

                cont_params.add(prefix + 'amp_BE', value=1e-10, min=0.0,
                                max=1e-5, vary=True)

                cont_params.add(prefix + 'T_e', value=15000, min=10000,
                                max=20000, vary=False)
                cont_params.add(prefix + 'tau_BE', value=1.0, min=0.1, max=2.0,
                                vary=False)

                cont_params.add(prefix + 'lambda_BE', value=3646, vary=False)

                cont_model = Model(power_law_at_2500A_plus_manual_BC,
                                   prefix=prefix)


            elif cont_model_name == "Balmer Continuum":

                if hasattr(self, 'redshift'):
                    redsh_guess = self.redshift
                else:
                    redsh_guess = 3.0

                cont_params = Parameters()

                cont_params.add('z', value=redsh_guess, min=0, max=1080,
                                vary=False)

                cont_params.add(prefix + 'z', value=redsh_guess, min=0,
                                max=1080, vary=False, expr='z')

                cont_params.add(prefix+'flux_BE', value=1, min=0, max=10, vary=True)
                cont_params.add(prefix+'T_e', value=15000, min=10000, max=20000, vary=True)
                cont_params.add(prefix+'tau_BE', value=0.5, min=0.1, max=2.0, vary=True)
                cont_params.add(prefix+'lambda_BE', value=3646, vary=False)

                cont_model = Model(balmer_continuum_model, prefix=prefix)

            elif cont_model_name == "Tsuzuki06":
                templ_fname = "Tsuzuki06.txt"
                cont_model, cont_params = load_template_model(
                    template_filename=templ_fname,
                    prefix=prefix)

            elif cont_model_name == "Vestergaard 01":
                templ_fname = "Fe_UVtemplt_A.asc"
                cont_model, cont_params = load_template_model(
                    template_filename=templ_fname,
                    prefix=prefix)


            elif cont_model_name == "Simqso Fe template":
                templ_fname = 'Fe_UVOPT_V01_T06_BR92.asc'

                if self.redshift is not None:
                    cont_model, cont_params = load_template_model(
                        template_filename=templ_fname,
                        prefix=prefix,
                        redshift=self.redshift,
                        flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = load_template_model(
                        template_filename=templ_fname,
                        prefix=prefix)


            elif cont_model_name == "Full subdivided iron template":

                if self.redshift is not None:
                    cont_model, cont_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                    templ_list=['UV01', 'UV02', 'UV03', 'UV04', 'UV05',
                                'UV06', 'OPT01', 'OPT02', 'OPT03'])


            elif cont_model_name == "Iron template 1200-2200":

                if self.redshift is not None:
                    cont_model, cont_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['UV01', 'UV02', 'UV03', 'UV03_FeIII'])

            elif cont_model_name == "Iron template 2200-3500 (T06)":

                if self.redshift is not None:
                    cont_model, cont_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['UV04', 'UV05', 'UV06'])

            elif cont_model_name == "Iron template 2200-3500 (V01)":

                if self.redshift is not None:
                    cont_model, cont_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['UV04_V01', 'UV05_V01', 'UV06_V01'])

            elif cont_model_name == "Iron template 3700-5600":

                if self.redshift is not None:
                    cont_model, cont_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['OPT01', 'OPT02', 'OPT03'])

            elif cont_model_name == "Iron template 2200-3500 (T06, cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_MgII(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_MgII()

            elif cont_model_name == "Iron template 2200-3500 (V01, cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_MgII_V01(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_MgII_V01()

            elif cont_model_name == "Iron template 2200-3500 new (T06, cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_MgII_new(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_MgII_new()

            elif cont_model_name == "Iron template 2200-3500 new (V01, cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_MgII_V01_new(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_MgII_V01_new()

            elif cont_model_name == "Iron template 1200-2200 (cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_CIV(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_CIV()

            elif cont_model_name == "Iron template 3700-5600 (cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_Hb(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_Hb()

            elif cont_model_name == "Iron template 3700-5600 new (cont.)":

                if self.redshift is not None:
                    cont_model, cont_params = iron_template_Hb_new(
                        redshift=self.redshift, flux_2500=self.flux_2500)
                else:
                    cont_model, cont_params = iron_template_Hb_new()

            if isinstance(cont_model, list) and isinstance(cont_params, list):
                self.cont_model_list.extend(cont_model)
                self.cont_model_par_list.extend(cont_params)

            else:
                self.cont_model_list.append(cont_model)
                self.cont_model_par_list.append(cont_params)

            self.build_full_cont_model()

            # Delete all hboxes in the list to rebuild them
            for hbox in self.cont_fit_hbox_list:
                self.deleteItemsOfLayout(hbox)
                self.remove_last_layout(hbox)
            # Rebuild all hboxes based on the new continuum model lists
            self.set_hbox_cont_fit()
            self.restore_hbox_cont_fit()


    def build_full_cont_model(self):
        """ Build the full continuum model from the models and parameters
        in the continnum model and parameter lists.

        This is the model that is fit.

        """

        self.cont_model_pars = Parameters()

        # For all model parameters in the parameter model list
        for idx, params in enumerate(self.cont_model_par_list):
            # For all parameters within the model parameters
            for jdx, p in enumerate(params):
                # Add the parameters to the continuum parameters
                self.cont_model_pars.add(p, expr=params[p].expr,
                                         value=params[p].value,
                                         min=params[p].min,
                                         max=params[p].max,
                                         vary=params[p].vary)

        # Add the redshift parameter if included in the fit
        if self.cont_fit_z_flag:
            self.cont_model_pars.add(self.redsh_par['z'],
                                     value=self.redsh_par['z'].value,
                                     min=self.redsh_par['z'].min,
                                     max=self.redsh_par['z'].max,
                                     vary=self.redsh_par['z'].vary)


        # Build the continuum model from the individual models
        if len(self.cont_model_list) > 0:
            self.cont_model = self.cont_model_list[0]

            for cont_model in self.cont_model_list[1:]:
                self.cont_model += cont_model

            # Calculate the initial fit
            spec = self.in_dict['spec'].copy()
            init_fit_flux = self.cont_model.eval(self.cont_model_pars,
                                                 x=spec.dispersion)
            self.in_dict["cont_init_spec"] = \
                sod.SpecOneD(dispersion=spec.dispersion,
                             flux=init_fit_flux, unit='f_lam')

        self.redraw()

    def delete_selected_cont_model(self):
        """ Delete a selected model from the continuum model.

        """

        # Get the index of the selected continuum model
        idx = self.cont_del_model_box.currentIndex()

        # Delete selected continuum model
        if len(self.cont_model_list) > 0:

            del self.cont_model_list[idx]
            del self.cont_model_par_list[idx]
            self.deleteItemsOfLayout(self.cont_fit_hbox_list[idx])
            del self.cont_fit_hbox_list[idx]

        if len(self.cont_model_par_list) > 0:
            self.build_full_cont_model()

        self.upd_cont_del_box()


    def remove_last_cont_from_model(self):
        """ Remove the last added model from the continuum model.

        """

        if len(self.cont_model_list) > 0:
            del self.cont_model_list[-1]
            del self.cont_model_par_list[-1]

            self.deleteItemsOfLayout(self.cont_fit_hbox_list[-1])

            del self.cont_fit_hbox_list[-1]

        if len(self.cont_model_par_list) > 0 :
            self.build_full_cont_model()

        self.upd_cont_del_box()


    def fit_cont_model(self):
        """ Fit the full continuum model using least-squares.

        """

        self.build_full_cont_model()

        spec = self.in_dict["spec"]
        m = np.logical_and(self.in_dict["mask_list"][1], spec.mask)

        if self.fit_with_weights:
            print(spec.flux[np.isnan(spec.flux)])
            weights = 1./spec.flux_err[m]**2
            print(weights[np.isnan(weights)])
            # Fitting with weights
            self.cont_fit_result = self.cont_model.fit(spec.flux[m],
                                                       self.cont_model_pars,
                                                       x=spec.dispersion[m],
                                                       weights=1./spec.flux_err[
                                                           m]**2)

        else:
            self.cont_fit_result = self.cont_model.fit(spec.flux[m],
                                                       self.cont_model_pars,
                                                       x=spec.dispersion[m])

        self.in_dict["cont_result"] = self.cont_fit_result

        cont_fit_flux = self.cont_model.eval(self.cont_fit_result.params, x=spec.dispersion)
        cont_fit_spec = sod.SpecOneD(dispersion=spec.dispersion,
                                     flux=cont_fit_flux, unit='f_lam')

        self.in_dict["cont_fit_spec"] = cont_fit_spec

        self.upd_cont_param_values_from_fit()
        self.upd_linedit_values_from_cont_params()

        self.redraw()

# ------------------------------------------------------------------------------
# LINE MODEL FIT FUNCTIONS
# ------------------------------------------------------------------------------

    def set_line_fit_box(self):
        """Create the vertical gui box for the line model mode.

        """

        self.line_fit_vbox = QVBoxLayout()
        self.line_fit_vbox_lines = QHBoxLayout()

        line_fit_groupbox = QGroupBox()
        line_fit_groupbox_layout = QVBoxLayout(line_fit_groupbox)

        line_model_list = ['M: Gaussian (FWHM)',
                           # 'M: Voigt Profile',
                           # 'M: QSO 4500-5500A (simple)',
                           # 'M: QSO 4500-5000A (complex)',
                           # 'M: QSO 2500-3500A',
                           'Line model MgII (1G)',
                           'Line model MgII (2G)',
                           'Line model CIV (1G)',
                           'Line model CIV (2G)',
                           'Line model CIII complex (3G)',
                           'Line model CIII complex (1G)',
                           "Line model Hbeta (6G)",
                           "Line model Hbeta (4G)",
                           "Line model Hbeta (2G)",
                           "Line model Halpha (3G)",
                           "Line model Halpha (2G)",
                           'Line model HeII (1G)',
                           'Line model SiIV (1G)',
                           'Iron template CIV',
                           'Iron template MgII',
                           'Iron template Hbeta',
        ]

        self.line_fit_box = QComboBox()
        for line_model in line_model_list:
            self.line_fit_box.addItem(line_model)


        self.prefix_label = QLabel("Prefix")
        self.prefix_in = QLineEdit('{}'.format("l"))
        self.prefix_in.setMaxLength(12)

        self.add_button = QPushButton("+")
        self.add_button.clicked.connect(self.add_line_to_model)
        self.remove_button = QPushButton("-")
        self.remove_button.clicked.connect(self.remove_last_line_from_model)
        self.line_del_model_box = QComboBox()
        self.line_del_model_button = QPushButton("Delete model")
        self.line_del_model_button.clicked.connect(self.delete_selected_line_model)

        self.fluxerrweights_line_checkbox = QCheckBox('Fit with weights (flux '
                                                'errors)')
        self.fluxerrweights_line_checkbox.setChecked(self.fit_with_weights)
        self.fluxerrweights_line_checkbox.stateChanged.connect(
            self.upd_line_values)

        for w in [self.line_fit_box,
                  self.prefix_label,
                  self.prefix_in,
                  self.add_button,
                  self.remove_button,
                  self.line_del_model_box,
                  self.line_del_model_button,
                  self.fluxerrweights_line_checkbox]:

            line_fit_groupbox_layout.addWidget(w)
            line_fit_groupbox_layout.setAlignment(w,  QtCore.Qt.AlignVCenter)

        ScArea = QScrollArea()
        ScArea.setLayout(QVBoxLayout())
        ScArea.setWidgetResizable(True)
        ScArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        ScArea.setWidget(line_fit_groupbox)

        self.line_fit_vbox.addWidget(ScArea)
        self.line_fit_hbox = QHBoxLayout()
        self.line_fit_hbox.addLayout(self.line_fit_vbox,1)
        self.line_fit_hbox.addLayout(self.line_fit_vbox_lines,2)

        self.layout.addLayout(self.line_fit_hbox)

        self.line_fit_z_lineditlist = []
        self.line_fit_z_varyboxlist = []

        self.set_hbox_line_fit()
        self.add_hbox_redshift()

        # Restore old models/parameters if they exist
        if self.line_model_par_list is not None and self.line_model_list is not None:
            self.set_hbox_line_fit()
            self.restore_hbox_line_fit()


    def add_hbox_redshift(self):
        """Add the redshift parameter box to the GUI.

        """

        label = QLabel('z')
        linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].value))
        linedit.setMaxLength(20)
        min_label = QLabel("min")
        min_linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].min))
        min_linedit.setMaxLength(20)
        max_label = QLabel("max")
        max_linedit = QLineEdit('{:.5f}'.format(self.redsh_par['z'].max))
        max_linedit.setMaxLength(20)
        vary_checkbox = QCheckBox("vary")
        vary_checkbox.setChecked(self.redsh_par['z'].vary)
        fit_z_checkbox = QCheckBox("fit z (overrides model z parameters)")
        fit_z_checkbox.setChecked(self.line_fit_z_flag)

        widgetlist = [label, linedit, min_label, min_linedit,
                      max_label, max_linedit, vary_checkbox, fit_z_checkbox]

        self.line_fit_z_lineditlist = [linedit, min_linedit, max_linedit]
        self.line_fit_z_varyboxlist = [vary_checkbox, fit_z_checkbox]

        for l in [linedit, min_linedit, max_linedit]:
            l.returnPressed.connect(self.upd_line_values)

        vary_checkbox.stateChanged.connect(self.upd_line_values)
        fit_z_checkbox.stateChanged.connect(self.upd_line_values)

        hbox_redshift = QVBoxLayout()
        for w in widgetlist:
            hbox_redshift.addWidget(w)
            hbox_redshift.setAlignment(w,  QtCore.Qt.AlignVCenter)

        self.line_fit_vbox.addLayout(hbox_redshift)


    def set_hbox_line_fit(self):
        """ Initialize the line model/parameter lists.

        """
        self.line_fit_hbox_widgetlist = []
        self.line_fit_lineditlist = []
        self.line_fit_varybox_list = []
        self.line_fit_hbox_list = []

    def upd_line_del_box(self):
        """ Update the delete drop down box with current line models

        """

        # Populates the box that allows to delete models
        self.line_del_model_box.clear()
        for idx, model in enumerate(self.line_model_list):
            self.line_del_model_box.addItem(model.prefix)

    def restore_hbox_line_fit(self):
        """ Restore the line fit hbox from the main menu or after loading
        a previous fit.

        """

        self.upd_line_del_box()

        for idx, params in enumerate(self.line_model_par_list):
            widgetlist = []
            lineditlist = []
            varyboxlist = []

            for param in params:
                label = QLabel(param)
                linedit = QLineEdit('{:.4E}'.format(params[param].value))
                linedit.setMaxLength(20)
                expr_linedit = QLineEdit('{}'.format(params[param].expr))
                expr_linedit.setMaxLength(20)
                min_label = QLabel("min")
                min_linedit = QLineEdit('{:.4E}'.format(params[param].min))
                min_linedit.setMaxLength(20)
                max_label = QLabel("max")
                max_linedit = QLineEdit('{:.4E}'.format(params[param].max))
                max_linedit.setMaxLength(20)
                vary_checkbox = QCheckBox("vary")
                vary_checkbox.setChecked(params[param].vary)

                widgetlist.extend([label, linedit, expr_linedit, min_label, min_linedit, max_label, max_linedit, vary_checkbox])
                lineditlist.extend([linedit, expr_linedit, min_linedit, max_linedit])
                varyboxlist.append(vary_checkbox)

                vary_checkbox.stateChanged.connect(self.upd_line_values)

            self.line_fit_hbox_widgetlist.extend(widgetlist)
            self.line_fit_lineditlist.append(lineditlist)
            self.line_fit_varybox_list.append(varyboxlist)

            line_groupbox = QGroupBox()
            layout_groupbox = QVBoxLayout(line_groupbox)
            vbox = QVBoxLayout()

            for w in widgetlist:
                layout_groupbox.addWidget(w)
                layout_groupbox.setAlignment(w,  QtCore.Qt.AlignVCenter)

            ScArea = QScrollArea()
            ScArea.setLayout(QVBoxLayout())
            ScArea.setWidgetResizable(True)
            ScArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

            ScArea.setWidget(line_groupbox)
            vbox.addWidget(ScArea)

            self.line_fit_hbox_list.append(vbox)
            self.line_fit_vbox_lines.addLayout(vbox)

            for l in lineditlist:
                l.returnPressed.connect(self.upd_line_values)

    def upd_line_values(self):
        """ Update the parameter values for all line models from the
        line input of the GUI.

        """
        # Update fit with weights
        # Update fit with errors
        self.fit_with_weights = self.fluxerrweights_line_checkbox.isChecked()

        # Update redshift parameter + fit flag
        new_z = float(self.line_fit_z_lineditlist[0].text())
        new_z_min = float(self.line_fit_z_lineditlist[1].text())
        new_z_max =float(self.line_fit_z_lineditlist[2].text())
        new_z_vary = self.line_fit_z_varyboxlist[0].isChecked()
        new_z_fit = self.line_fit_z_varyboxlist[1].isChecked()

        self.redsh_par['z'].set(value=new_z, min=new_z_min,
                                max=new_z_max, vary=new_z_vary)
        self.line_fit_z_flag = new_z_fit

        # Update all parameters in line_model_par_list
        for idx, line_model_pars in enumerate(self.line_model_par_list):

            params = self.line_model_par_list[idx]
            build_model = True

            for jdx, param in enumerate(params):

                try:
                    new_value = float(self.line_fit_lineditlist[idx][jdx*4+0].text())
                    new_expr = self.line_fit_lineditlist[idx][jdx*4+1].text()
                    new_min = float(self.line_fit_lineditlist[idx][jdx*4+2].text())
                    new_max = float(self.line_fit_lineditlist[idx][jdx*4+3].text())
                    new_vary = self.line_fit_varybox_list[idx][jdx].isChecked()

                    if new_expr == 'None':
                        new_expr = None

                    self.line_model_par_list[idx][param].set(value=new_value, expr=new_expr, min=new_min, max=new_max, vary=new_vary)

                except:
                    print("Input does not conform to string or float limitations!")
                    build_model = False

            if build_model:
                self.build_full_line_model()


    def upd_line_param_values_from_fit(self):
        """Update the line model parameters from the latest best-fit
        continuum model.

        """

        # Update redshift parameter
        if self.line_fit_z_flag:
            temp_val = self.line_fit_result.params['z'].value
            self.redsh_par['z'].value = temp_val

        # Update all parameters in line_model_par_list
        for idx, line_model in enumerate(self.line_model_list):

            params = self.line_model_par_list[idx]

            for jdx, param in enumerate(params):

                temp_val = self.line_fit_result.params[param].value
                self.line_model_par_list[idx][param].value = temp_val

                temp_val = self.line_fit_result.params[param].expr
                self.line_model_par_list[idx][param].expr = temp_val

                temp_val = self.line_fit_result.params[param].min
                self.line_model_par_list[idx][param].min = temp_val

                temp_val = self.line_fit_result.params[param].max
                self.line_model_par_list[idx][param].max = temp_val

                temp_val = self.line_fit_result.params[param].vary
                self.line_model_par_list[idx][param].vary = temp_val



    def upd_linedit_values_from_line_params(self):
        """ Update the line edit values based on the current line model
        parameters.

        """

        # Update the redshift parameter
        if self.line_fit_z_flag:
            temp_val = self.redsh_par['z'].value
            self.line_fit_z_lineditlist[0].setText("{0:.4f}".format(temp_val))

        # Update all parameters in line_model_par_list
        for idx, line_model in enumerate(self.line_model_list):
            params = self.line_model_par_list[idx]
            print(line_model, params, self.line_fit_lineditlist)

            for jdx, param in enumerate(params):
                temp_val = self.line_model_par_list[idx][param].value
                self.line_fit_lineditlist[idx][jdx*4+0].setText("{"
                                                                "0:.4E}".format(temp_val))
                temp_val = self.line_model_par_list[idx][param].expr
                self.line_fit_lineditlist[idx][jdx*4+1].setText("{}".format(temp_val))

    def on_press_line_fit(self, event):
        """ Set the key functionality in the "line model" mode

            Parameters
            ----------
            event : key_press_event
                Key press event to evaluate
        """

        self.last_on_press = self.on_press_line_fit
        self.last_help_label_function = self.line_fit_help_labels

        if event.key == "q":
            # Quit line model mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.deleteItemsOfLayout(self.line_fit_hbox)
            self.remove_last_layout(self.line_fit_hbox)
            self.gcid = self.specfitcanvas.mpl_connect('key_press_event',
                                                       self.on_press_main)
            self.statusBar().showMessage("SpecFitGui Main Mode", 5000)
            self.update_help_window(self.main_help_labels)

        elif event.key == "e":
            # Change to dispersion zoom mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.statusBar().showMessage("Mode: Set wavelength "
                                         "range to display", 5000)
            self.update_help_window(self.ranges_help_labels)
            self.handleStart(self.on_press_line_fit)

        elif event.key == "m":
            # Change to masking mode
            self.specfitcanvas.mpl_disconnect(self.gcid)
            self.set_mask_hbox()
            self.statusBar().showMessage("Masking mode", 5000)
            self.cid = self.specfitcanvas.mpl_connect('key_press_event',
                                                      self.on_press_mask)
            self.update_help_window(self.mask_help_labels)

        elif event.key == "f":
            # Fit line model
            self.fit_line_model()

        elif event.key == "S":
            # Save fit in folder
            self.save_fit()

        elif event.key == "?":
            # Toggle help window
            self.help_window_toggle(self.line_fit_help_labels)

    def line_fit_help_labels(self):
        """ Set the labels for the help window

        """

        l1 = QLabel("Hot Keys - Line Fitting")
        l1.setAlignment(QtCore.Qt.AlignHCenter)

        l2 = QLabel("e : Change displayed wavelength range")
        l2.setAlignment(QtCore.Qt.AlignHCenter)
        l2.setWordWrap(True)

        l3 = QLabel("m : Start masking mode")
        l3.setAlignment(QtCore.Qt.AlignHCenter)
        l3.setWordWrap(True)

        l4 = QLabel("f : Fit line model")
        l4.setAlignment(QtCore.Qt.AlignHCenter)
        l4.setWordWrap(True)

        l5 = QLabel("S : Save fit")
        l5.setAlignment(QtCore.Qt.AlignHCenter)
        l5.setWordWrap(True)

        l6 = QLabel("q : Quit")
        l6.setAlignment(QtCore.Qt.AlignHCenter)
        l6.setWordWrap(True)

        l7 = QLabel("? : Toggle hot key info")
        l7.setAlignment(QtCore.Qt.AlignHCenter)
        l7.setWordWrap(True)

        return [l1, l2, l3, l4, l5, l6, l7]


    def add_line_to_model(self):
        """ Add a model to the line model.

         """

        prefix = self.prefix_in.text()+'_'
        prefix_flag = True
        line_model_name = self.line_fit_box.currentText()

        # Check if prefix is already used in line model
        for model in self.line_model_list:
            if prefix == model.prefix or \
                    ((line_model_name=="M: QSO 4500-5500A (simple)" or
                      line_model_name=="M: QSO 4500-5500A (complex)")
                     and model.prefix =='hbeta_b_') or \
                    (line_model_name == "M: QSO 2500-3500A"
                     and model.prefix == 'mgII_b_'):
                self.statusBar().showMessage("Line model with same prefix"
                                             " exists already! Please choose"
                                             " a different prefix.", 5000)
                prefix_flag = False

        # Add model to the line model, if the prefix has not been used before
        if prefix_flag:

            if self.line_fit_z_flag:
                pars = self.redsh_par.copy()
            else:
                pars = None

            if line_model_name == "M: Gaussian (FWHM)":
                line_params, line_model = \
                    emission_line_model(amp=20,
                                        cen=4861,
                                        wid=2000,
                                        shift=0,
                                        unit_type="fwhm_km_s",
                                        prefix=prefix,
                                        fit_central=True,
                                        parameters=pars)

                if self.line_fit_z_flag:
                    line_params.add('z', value=self.redshift, min=self.redshift * 0.9,
                               max=max(self.redshift * 1.1, 1),
                               vary=True)
                else:
                    line_params.add('z', value=self.redshift, min=self.redshift * 0.9,
                               max=max(self.redshift * 1.1, 1),
                               vary=False)

                # self.line_model_list.append(line_model)
                # self.line_model_par_list.append(line_params)

            # elif line_model_name == "M: QSO 4500-5500A (simple)":
            #
            #     line_params, line_model = \
            #         create_qso_model_4500_5500(fit_z=self.line_fit_z_flag,
            #                                    redsh=self.redshift)

                # self.line_model_list.extend(model_list)
                # self.line_model_par_list.extend(param_list)

            # elif line_model_name == "M: QSO 4500-5500A (complex)":
            #
            #     line_params, line_model = \
            #         create_qso_model_4500_5500_complex(
            #             fit_z=self.line_fit_z_flag,
            #                                    redsh=self.redshift)

                # self.line_model_list.extend(model_list)
                # self.line_model_par_list.extend(param_list)

            # elif line_model_name == "M: QSO 2500-3500A":
            #
            #     line_params, line_model = \
            #         create_qso_model_2500_3500(
            #             fit_z=self.line_fit_z_flag,
            #                                    redsh=self.redshift)

            elif line_model_name == "Line model MgII (1G)":

                line_params, line_model = \
                    create_line_model_MgII_1G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "Line model MgII (2G)":

                line_params, line_model = \
                    create_line_model_MgII_2G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "Line model CIV (2G)":

                line_params, line_model = \
                    create_line_model_CIV_2G(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)

            elif line_model_name == "Line model CIV (1G)":

                line_params, line_model = \
                    create_line_model_CIV_1G(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)

            elif line_model_name == "Line model CIII complex (3G)":

                line_params, line_model = \
                    create_line_model_CIII(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)

            elif line_model_name == "Line model CIII complex (1G)":

                line_params, line_model = \
                    create_line_model_CIII_1G(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)


            elif line_model_name == "Line model HeII (1G)":

                line_params, line_model = \
                    create_line_model_HeII_HighZ(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)

            elif line_model_name == "Line model SiIV (1G)":

                line_params, line_model = \
                    create_line_model_SiIV_HighZ(fit_z=self.line_fit_z_flag,
                                           redsh=self.redshift)

            elif line_model_name == "Line model Hbeta (6G)":

                line_params, line_model = \
                    create_line_model_HbOIII_6G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "Line model Hbeta (4G)":

                line_params, line_model = \
                    create_line_model_HbOIII_4G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)


            elif line_model_name == "Line model Hbeta (2G)":

                line_params, line_model = \
                    create_line_model_Hb_2G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "Line model Halpha (2G)":
                line_params, line_model = \
                    create_line_model_Ha_2G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "Line model Halpha (3G)":
                line_params, line_model = \
                    create_line_model_Ha_3G(fit_z=self.line_fit_z_flag,
                                               redsh=self.redshift)

            elif line_model_name == "M: Voigt Profile":
                line_model = VoigtModel(prefix=prefix)

                line_params = Parameters()
                line_params.update(line_model.make_params())



            elif line_model_name == "Iron template CIV":

                if self.redshift is not None:
                    line_model, line_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['UV01', 'UV02'])

            elif line_model_name == "Iron template MgII":

                if self.redshift is not None:
                    line_model, line_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['UV05'])

            elif line_model_name == "Iron template Hbeta":

                if self.redshift is not None:
                    line_model, line_params = subdivided_iron_template(
                        redshift=self.redshift, flux_2500=self.flux_2500,
                        templ_list=['OPT02'])

            if isinstance(line_model, list) and isinstance(line_params, list):
                self.line_model_list.extend(line_model)
                self.line_model_par_list.extend(line_params)

            else:
                self.line_model_list.append(line_model)
                self.line_model_par_list.append(line_params)

            # Add the new model to the line model
            self.build_full_line_model()

            # Delete all line model hboxes to rebuild them
            for hbox in self.line_fit_hbox_list:
                self.deleteItemsOfLayout(hbox)
                self.remove_last_layout(hbox)
            self.set_hbox_line_fit()
            # Rebuild all hboxes based on the new continuum model lists
            self.restore_hbox_line_fit()

    def build_full_line_model(self):
        """ Build the full line model from the models and parameters
        in the line model and parameter lists.

        This is the model that is fit.

        """

        self.line_model_pars = Parameters()

        # For each set of model parameters in the model parameter set list
        for idx, params in enumerate(self.line_model_par_list):
            # For each parameter in the model parameter set
            for jdx, p in enumerate(params):
                # Add parameter to full line model parameters
                self.line_model_pars.add(p, expr=params[p].expr,
                                         value=params[p].value,
                                         min=params[p].min,
                                         max=params[p].max,
                                         vary=params[p].vary)

        # Add the redshift to the fit, if selected
        if self.line_fit_z_flag:
            self.line_model_pars.add(self.redsh_par['z'],
                                     value=self.redsh_par['z'].value,
                                     min=self.redsh_par['z'].min,
                                     max=self.redsh_par['z'].max,
                                     vary=self.redsh_par['z'].vary)

        # Add all models in the model list to the full line model
        if len(self.line_model_list) > 0:
            self.line_model = self.line_model_list[0]

            for line_model in self.line_model_list[1:]:
                self.line_model += line_model

            # Calculate the initial line model fit
            spec = self.in_dict["spec"].copy()
            init_fit_flux = self.line_model.eval(self.line_model_pars,
                                                 x=spec.dispersion)
            self.in_dict["line_init_spec"] = \
                sod.SpecOneD(dispersion=spec.dispersion,
                             flux=init_fit_flux, unit='f_lam')

        self.redraw()

    def delete_selected_line_model(self):
        """ Delete a selected model from the continuum model.

        """


        # Get index of the selected model
        idx = self.line_del_model_box.currentIndex()

        print('IDX to delete', idx)
        print(self.line_model_list)

        # Delete selected model from the list
        if len(self.line_model_list) > 0:

            del self.line_model_list[idx]
            del self.line_model_par_list[idx]
            self.deleteItemsOfLayout(self.line_fit_hbox_list[idx])
            del self.line_fit_hbox_list[idx]

        if len(self.line_model_par_list) > 0:
            self.build_full_line_model()

        print(self.line_model_list)

        # Rebuild the GUI accordingly
        self.upd_line_del_box()


    def remove_last_line_from_model(self):
        """ Remove the last added model from the line model.

         """

        if len(self.line_model_list) > 0:
            del self.line_model_list[-1]
            del self.line_model_par_list[-1]

            self.deleteItemsOfLayout(self.line_fit_hbox_list[-1])

            del self.line_fit_hbox_list[-1]

        if len(self.line_model_par_list) > 0 :
            self.build_full_line_model()

        self.upd_line_del_box()



    def fit_line_model(self):
        """ Fit the full continuum model using least-squares.

        """

        self.build_full_line_model()

        spec = self.in_dict["spec"]
        cont_spec = self.in_dict["cont_fit_spec"]
        # Calculate the continuum subtracted spectrum
        cont_resid = spec.subtract(cont_spec)
        m = np.logical_and(self.in_dict["mask_list"][2], spec.mask)

        if self.fit_with_weights:
            # Fitting with weights
            self.line_fit_result = self.line_model.fit(cont_resid.flux[m],
                                                       self.line_model_pars,
                                                       x=cont_resid.dispersion[m],
                                                       weights=1./spec.flux_err[
                                                           m]**2)

        else:
            self.line_fit_result = self.line_model.fit(cont_resid.flux[m],
                                                       self.line_model_pars,
                                                       x=cont_resid.dispersion[m])



        self.in_dict["line_result"] = self.line_fit_result

        line_fit_flux = self.line_model.eval(self.line_fit_result.params, x=cont_resid.dispersion)
        line_fit_spec = sod.SpecOneD(dispersion=cont_resid.dispersion, flux=line_fit_flux, unit='f_lam')
        self.in_dict["line_fit_spec"] = line_fit_spec

        self.upd_line_param_values_from_fit()
        self.upd_linedit_values_from_line_params()

        self.redraw()

# ------------------------------------------------------------------------------
# EVALUATION MODE FUNCTIONS (NEW, NOT IMPLEMENTED YET)
# ------------------------------------------------------------------------------

    def set_qso_eval_box(self):

        # make a hbox that allows for tabs
        # two tabs at the beginning 1) continuum eval 2) line eval

        pass

    def on_press_qso_eval(self):

        pass

# ------------------------------------------------------------------------------
# TEST FUNCTIONS (MAY BE NOT UP TO DATE)
# ------------------------------------------------------------------------------


def test_specfit():

    # data = np.genfromtxt('J172+18/P172+18ldss3prelim_Jntt.spc')
    # data = np.genfromtxt('J1152/J1152_calib_NIR.dat')
    # data = np.genfromtxt("vandenberk2001.dat")
    # spec = sod.SpecOneD(dispersion=data[:,0], flux=data[:,1], flux_err=data[
    #                                                                    :,2], unit='f_lam')
    df = pd.read_hdf('J2125-1719_composite_smooth.hdf5','data')
    spec = sod.SpecOneD(dispersion=df['dispersion'].values, flux = df[
        'flux'].values, flux_err = df['flux_err'], unit='f_lam')
    spec.redshift
    app = QtWidgets.QApplication(sys.argv)
    form = SpecFitGui(spec=spec, redshift=3.9)
    form.show()
    app.exec_()


def test_single_emission_line_fit():

    ly_alpha_params, ly_alpha = emission_line_model(amp=20, cen=1215, wid=2000,
                                                    shift=-10,
                                                    unit_type="fwhm_km_s",
                                                    prefix='lya_',
                                                    fit_central=False)

    print(ly_alpha.param_names, ly_alpha.independent_vars)
    print(ly_alpha_params)

    x = np.arange(800,1400,5)

    y = gaussian_fwhm_km_s(x, 30, 1215, 2500, 20)

    result = ly_alpha.fit(y, ly_alpha_params, x=x)

    print(result.fit_report())


def test_multiple_redshift_dependent_emission_line_fit():

    pars = Parameters()

    pars.add('redsh',value=3.1, vary = True, min=2.8, max=3.5)


    params, hbeta = emission_line_model(amp=20, cen='(redsh+1)*4861',
                                                          wid=2000,
                                                          shift=-10,
                                                          unit_type="fwhm_km_s",
                                                          prefix='hbeta_',
                                                          fit_central=True,
                                                          parameters=pars)

    params, OIII_a = emission_line_model(amp=20, cen='(redsh+1)*4959',
                                                          wid=500,
                                                          shift='hbeta_shift_km_s',
                                                          unit_type="fwhm_km_s",
                                                          prefix='OIIIa_',
                                                          fit_central=True,
                                                          parameters=params)

    params_b, OIII_b = emission_line_model(amp=20, cen='(redsh+1)*5007',
                                                          wid=500,
                                                          shift='hbeta_shift_km_s',
                                                          unit_type="fwhm_km_s",
                                                          prefix='OIIIb_',
                                                          fit_central=True,
                                                          parameters=pars)

    model = hbeta + OIII_a + OIII_b


    params['hbeta_shift_km_s'].set(min=-200, max=200)
    params['OIIIa_shift_km_s'].set(min=-200, max=200)
    params['OIIIb_shift_km_s'].set(min=-200, max=200)

    params['hbeta_amp'].set(min=0)
    params['OIIIa_amp'].set(min=0)
    params['OIIIb_amp'].set(min=0)

    # pars.add(OIII_params_a)

    x = np.arange(4000,6000,1) * (3.123 + 1)

    z = 3.123 + 1

    y = gaussian_fwhm_km_s(x, 20, z*4861, 3500, 10) + gaussian_fwhm_km_s(x, 20, z*4959, 1000, 10) + gaussian_fwhm_km_s(x, 20, z*5007, 1000, 10)

    result = model.fit(y, params, x=x)

    print(model.param_names, model.independent_vars)
    print(params)

    plt.plot(x, result.init_fit,ls='-.')
    plt.plot(x,y)
    plt.plot(x, result.best_fit,ls='--')


    plt.show()

    print(result.fit_report())


def test_power_law_fit():

    pars = Parameters()

    pl_model = Model(power_law, prefix='cont_')

    pars.add('cont_amp', value=100, min=0, max=5000)
    pars.add('cont_slope', value = -1.5, min=-2.5, max=-0.5)

    x = np.arange(4000,6000,1)

    y = power_law(x, 1234, -1.3)

    result_cont = pl_model.fit(y, pars, x=x)

    print (result_cont.fit_report())


def test_emission_lines_and_power_law():

    pars = Parameters()

    pars.add('redsh',value=3.1, vary = True, min=2.8, max=3.5)


    pl_model = Model(power_law_continuum, prefix='cont_')

    pars.add('cont_amp', value=1000, min=0, max=5000)
    pars.add('cont_slope', value = -1.0, min=-5, max=-0.5)

    pl_pars = pars

    params, hbeta = emission_line_model(amp=20, cen='(redsh+1)*4861',
                                                          wid=2000,
                                                          shift=-10,
                                                          unit_type="fwhm_km_s",
                                                          prefix='hbeta_',
                                                          fit_central=True,
                                                          parameters=pars)

    params, OIII_a = emission_line_model(amp=20, cen='(redsh+1)*4959',
                                                          wid=500,
                                                          shift='hbeta_shift_km_s',
                                                          unit_type="fwhm_km_s",
                                                          prefix='OIIIa_',
                                                          fit_central=True,
                                                          parameters=params)

    params, OIII_b = emission_line_model(amp=20, cen='(redsh+1)*5007',
                                                          wid=500,
                                                          shift='hbeta_shift_km_s',
                                                          unit_type="fwhm_km_s",
                                                          prefix='OIIIb_',
                                                          fit_central=True,
                                                          parameters=params)

    model = pl_model + hbeta + OIII_a + OIII_b


    params['hbeta_shift_km_s'].set(min=-200, max=200)
    params['OIIIa_shift_km_s'].set(min=-200, max=200)
    params['OIIIb_shift_km_s'].set(min=-200, max=200)

    params['hbeta_amp'].set(min=0)
    params['OIIIa_amp'].set(min=0)
    params['OIIIb_amp'].set(min=0)

    # pars.add(OIII_params_a)

    x = np.arange(4000,6000,1) * (3.123 + 1)

    z = 3.123 + 1

    y = power_law_continuum(x, 1000, -1.1) + gaussian_fwhm_km_s(x, 20, z*4861, 3500, 10) + gaussian_fwhm_km_s(x, 20, z*4959, 1000, 10) + gaussian_fwhm_km_s(x, 20, z*5007, 1000, 10)

    result = model.fit(y, params, x=x)



    print(model.param_names, model.independent_vars)
    print(params)

    plt.plot(x, result.init_fit,ls='-.')
    plt.plot(x,y)
    plt.plot(x, result.best_fit,ls='--')


    plt.show()

    print(result.fit_report())


def test_vandenberk():

    data = np.genfromtxt("vandenberk2001.dat")
    spec = sod.SpecOneD(dispersion=data[:,0], flux=data[:,1],flux_err=data[:,2], unit='f_lam')

    spec.plot()


def test_expression_instead_of_value():

    p = Parameters()

    p.add('z',value=2,min=0,max=7,vary=True)

    pm, m = emission_line_model(amp=20, cen='4861',
                                wid=2000,
                                shift=-10,
                                unit_type="fwhm_km_s",
                                prefix='hbeta_',
                                fit_central=True,
                                parameters=p)

    print(pm,'\n')
    pm['hbeta_cen'].set(expr='(z+1)*4861')
    print(pm,'\n')
    pm['hbeta_cen'].set(value=4861, vary=True)
    print(pm,'\n')
    pm['hbeta_cen'].set(expr='(z+1)*4861')
    print(pm,'\n')
    # --->>> The dependent parameter needs to be part of the parameters


def test_template_model_fit():

    templ_fname = 'Tsuzuki06.txt'

    data = np.genfromtxt('iron_templates/'+templ_fname)

    to_fit = sod.SpecOneD(dispersion=data[:,0], flux=data[:,1]*1.3, unit='f_lam')

    to_fit.redshift(1.123, inplace=True)

    template_model, template_params = load_template_model(template_filename=templ_fname)

    template_params['redshift'].set(value=1.118, min=1.0, max=1.2)

    result = template_model.fit(to_fit.flux, template_params, x=to_fit.dispersion)

    plt.plot(to_fit.dispersion, to_fit.flux,'r-')
    plt.plot(to_fit.dispersion, result.init_fit,ls='-.',color='grey')
    # plt.plot(x,y)
    plt.plot(to_fit.dispersion, result.best_fit,ls='--',color='k')


    plt.show()

    print(result.fit_report())


def test_balmer_continuum(flux_BE=None, T_e=None, tau_BE=None, lambda_BE=3646):
    x = np.arange(500,3746,0.1)
    y = balmer_continuum_model(x, flux_BE, T_e, tau_BE, lambda_BE)

    plt.plot(x,y)
    plt.show()

# test_single_emission_line_fit()

# test_power_law_fit()

# test_multiple_redshift_dependent_emission_line_fit()

# test_emission_lines_and_power_law()

# test_balmer_continuum(flux_BE=1e-14, T_e=15000, tau_BE=1.)

# test_specfit()

# test_template_model_fit()

# test_expression_instead_of_value()

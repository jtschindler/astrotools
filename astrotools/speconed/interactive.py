"""Interactive GUI for 1D spectral manipulation.

This module provides interative GUIs to manipulate 1D spetroscopic data in
concert with the functions from the speconed.py module.


Example
-------
This is a short example on how to call the interactive GUI to determine the
velocity shift between two spectra.

    $ python example_numpy.py



Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.
"""

from __future__ import unicode_literals
import os
import numpy as np
import random
import time

import sys

from scipy import interpolate

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QLabel, QHBoxLayout, QLineEdit, QCheckBox
from PyQt5.QtGui import QIcon


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make sure that we are using QT5
# matplotlib.use('Qt5Agg')
# from PyQt5 import QtCore, QtWidgets
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from astrotools.speconed import speconed as sod
from astrotools.speconed.speconed import SpecOneD
# from astrotools.speconed.speconed import FlatSpectrum
# from astrotools.speconed.speconed import PassBand
from astrotools import constants as const

from numpy import arange, sin, pi

from lmfit import Model



def velocity_to_z(v):
    """ Calculates the redshift resulting from a doppler velocity (km/s).

    Parameters
    ----------
    v : float
        Doppler velocity in km/s

    Returns
    -------
    z : float
        Doppler redshift
    """

    beta = v*1000./const.c

    return np.sqrt((1+beta) / (1-beta)) - 1

def z_to_velocity(z):
    """ Calculates the the doppler velocity (km/s) of a doppler redshift.

    Parameters
    ----------
    z : float
        Doppler redshift

    Returns
    -------
    v : float
        Doppler velocity in km/s
    """

    beta = ((z+1)**2-1) / ((z+1)**2+1)

    return beta * const.c / 1000.



class SpecPlotCanvas(FigureCanvas):

    """A FigureCanvas for plotting up to two different spectra.

    This class provides the plotting routines for plotting a primary and a
    secondary spectrum.
    """

    def __init__(self, parent=None, in_dict=None, active=None, width=16, height=16, dpi=100):

        """__init__ method for the SpecPlotCanvas class

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        parent : obj, optional
            Parent class of SpecPlotCanvas
        width : float, optional
            The width of the matplotlib figure
        height : float, optional
            The height of the matplotlib figure
        dpi : float
            The dpi of the matplotlib figure


        """

        fig = Figure(figsize=(width,height), dpi=dpi)

        self.ax = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


        self.plot(in_dict, active)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


    def plot(self, in_dict, active=None):

        """Plotting function for the SpecPlotCancas class

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        active : int, optional
            Index that specifies the active spectrum in the spec_list within
            in_dict
        """

        self.ax.clear()

        # Setting up the inactive spectra
        for spec in in_dict['spec_list'][:active] + in_dict['spec_list'][active+1:]:

            spec_sec = spec
            # spec_sec.z = spec.z
            # spec_sec.yshift = spec.yshift
            mask_sec = spec.mask


            self.ax.plot(spec_sec.raw_dispersion,
                         spec_sec.raw_flux, 'grey', linewidth=1.5)
            self.ax.plot(spec_sec.redshift(spec_sec.z).dispersion[mask_sec],
                         spec_sec.flux[mask_sec]+spec_sec.yshift, 'k',
                         linewidth=1.5)

        # Setting up the active spectrum
        spec_prime = in_dict['spec_list'][active]
        # spec_prime.z = in_dict['spec_list'][active].z
        # spec_prime.yshift = in_dict['spec_list'][active].yshift
        mask_prime = spec_prime.mask


        # Plotting active spectrum
        self.ax.axhline(y=0.0, linewidth=1.5, color='k', linestyle='--')
        self.ax.plot(spec_prime.raw_dispersion,
                     spec_prime.raw_flux, 'grey', linewidth=2)
        self.ax.plot(spec_prime.redshift(spec_prime.z).dispersion[mask_prime],
                     spec_prime.flux[mask_prime]+spec_prime.yshift,
                     'r', linewidth=2)

        if spec_prime.fit_flux is not None:

            self.ax.plot(spec_prime.fit_dispersion, spec_prime.fit_flux,'b')


        # Setting the plot boundaries
        self.ax.set_xlim(in_dict['x_lo'], in_dict['x_hi'])
        self.ax.set_ylim(in_dict['y_lo'], in_dict['y_hi'])

        self.draw()

class ResultCanvas(FigureCanvas):

    """A FigureCanvas for plotting one spectrum as a result of an operation.

    This class provides the plotting routines for plotting a spectrum
    resulting from a multiplication of division of two spectra.
    """

    def __init__(self, parent=None, in_dict=None, mode=None, width=16, height=16, dpi=100):

        """__init__ method for the SpecPlotCanvas class

        Parameters
        ----------
        parent : obj, optional
            Parent class of SpecPlotCanvas
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data.
        mode : str,
            A string indicating the numerical operation: multiplication/
            division.
        width : float, optional
            The width of the matplotlib figure
        height : float, optional
            The height of the matplotlib figure
        dpi : float
            The dpi of the matplotlib figure


        """

        fig = Figure(figsize=(width,height), dpi=dpi)

        self.ax = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)


        self.result_plot(in_dict, mode)

        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def result_plot(self, in_dict, mode='divide'):

        """Plotting the result spectrum of a multiplication/division of two
        spectra.

        Parameters
        ----------
        in_dict : dictionary
            A dictionary containing the SpecOneD objects to be plotted as well
            as additional data
        mode : str
            A string that defines the mode of the action: 'multiply' or
            'divide'
        """

        self.ax.clear()

        # Setting up the primary spectrum
        spec_prime = in_dict['spec_list'][0].copy()
        spec_prime.z = in_dict['spec_list'][0].z
        spec_prime.yshift = in_dict['spec_list'][0].yshift
        spec_prime.redshift(spec_prime.z, inplace=True)

        # setting up the secondary spectrum
        spec_sec = in_dict['spec_list'][1].copy()
        spec_sec.z = in_dict['spec_list'][1].z
        spec_sec.yshift = in_dict['spec_list'][1].yshift
        spec_sec.redshift(spec_sec.z, inplace=True)

        # multiplication/divison of the spectra
        if mode == "divide":
            result_spec = spec_prime.divide(spec_sec)
        if mode == "multiply":
            result_spec = spec_prime.multiply(spec_sec)

        self.ax.plot(result_spec.dispersion[result_spec.mask],
                     result_spec.flux[result_spec.mask], 'k', lw=1.5)

        # Setting the plot boundaries
        x2_lo = in_dict['x2_lo']
        x2_hi = in_dict['x2_hi']
        lo_index = np.argmin(np.abs(result_spec.dispersion - x2_lo))
        hi_index = np.argmin(np.abs(result_spec.dispersion - x2_hi))
        median = np.median(result_spec.flux[lo_index : hi_index])
        std = np.std(result_spec.flux[lo_index : hi_index])

        self.ax.set_xlim(x2_lo, x2_hi)
        # self.ax.set_ylim(in_dict['y2_lo'], in_dict['y2_hi'])

        self.ax.set_ylim(median-3.5*std, median+3.5*std)

        self.draw()




class SpecOneDGui(QMainWindow):

    """The main interactive SpecOneD GUI.

    This class provides all interactive capabilities for the manipulation of
    one dimensional spectra (SpecOneD objects).

    Attributes
    ----------

    """



    def __init__(self, spec_list, mode=None):
        """__init__ method for the SpecOneDGui class

        Parameters
        ----------
        spec_list : list
            A list of SpecOneD objects to be manipulated
        mode : str
            A string specifying which functionality the GUI should provide.
            Currently implemented modes are:
        """

        QtWidgets.QMainWindow.__init__(self)

        self.act = 0

        for spec in spec_list:
            spec.z = 0
            spec.yshift = 0
            spec.scale = 1.

        self.shift = None
        self.dshift = None
        self.yshift = None

        self.mx1 = 0
        self.mx2 = 0

        self.spec_list_copy = spec_list

        self.in_dict = {'spec_list': spec_list,
                        'x_lo': None,
                        'x_hi': None,
                        'y_lo': None,
                        'y_hi': None}

        if mode == None or mode == "example":
            self.SimplePlotMode()
        elif mode == "divide":
            self.ResultPlotMode(mode)
        else:
            raise ValueError("Warning: Specified mode not known.")



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

    def remove_last_layout(self, layout):
        """This function removes the latest layout added to the GUI's general
        layout.

        Parameters
        ----------
        layout : obj, layout
            The general layout from which the latest (sub-)layout shall be
            removed.
        """

        # Remove mode specific layout and widgets
        for i in reversed(range(layout.count())):
            widgetToRemove = layout.itemAt(i).widget()
            # remove it from the layout list
            layout.removeWidget( widgetToRemove )
            # remove it from the gui
            widgetToRemove.setParent( None )
        self.layout.removeItem(self.layout.itemAt(-1))


    def ResultPlotMode(self, mode, normalize=True):

        """ This function activates the result plotting mode.

        Parameters
        ----------
        mode : str
            A string indicating the mode of the numerical operation for the
            two input spectra of the ResultPlotMode.
        normalize : boolean, optional
            Boolean that decides whether all spectra will be normalized to the
            zeroth spectrum in the sequence to present them on one scale.
        """
        self.main_widget = QtWidgets.QWidget(self)

        self.mode = mode


        print (self.act, self.act+1, self.in_dict['spec_list'])

        if normalize:
            for spec in self.in_dict['spec_list'][:self.act] + self.in_dict['spec_list'][self.act+1:]:

                spec.renormalize_by_spectrum(
                            self.in_dict['spec_list'][self.act],
                            trim_mode='wav',
                            inplace=True)





        # Calculating default plot limits
        self.in_dict['x_lo'] = min(self.in_dict['spec_list'][self.act].dispersion)
        self.in_dict['x_hi'] = max(self.in_dict['spec_list'][self.act].dispersion)
        self.in_dict['y_lo'] = min(self.in_dict['spec_list'][self.act].flux)
        self.in_dict['y_hi'] = max(self.in_dict['spec_list'][self.act].flux)

        self.in_dict['x2_lo'] = min(self.in_dict['spec_list'][self.act].dispersion)
        self.in_dict['x2_hi'] = max(self.in_dict['spec_list'][self.act].dispersion)
        if mode=="divide":
            result_flux = self.in_dict['spec_list'][0].divide(self.in_dict['spec_list'][1]).flux
            self.in_dict['y2_lo'] = min(result_flux[300:-300])
            self.in_dict['y2_hi'] = max(result_flux[300:-300])
        elif mode == "multiply":
            result_flux = self.in_dict['spec_list'][0].multiply(self.in_dict['spec_list'][1]).flux
            self.in_dict['y2_lo'] = min(self.in_dict['spec_list'][0].flux)
            self.in_dict['y2_hi'] = max(self.in_dict['spec_list'][0].flux)


        # Layout of the main GUI
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.tx1 = min(self.in_dict['spec_list'][self.act].dispersion)
        self.tx2 = max(self.in_dict['spec_list'][self.act].dispersion)

        self.specplotcanvas = SpecPlotCanvas(self, in_dict=self.in_dict, active=self.act)

        self.resultcanvas = ResultCanvas(self, in_dict=self.in_dict, mode=mode)

        # self.layout.addWidget(self.specplotcanvas)
        # self.layout.addWidget(self.resultcanvas)

        self.specplotcanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.resultcanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.specplotcanvas.setFocus()
        # self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)



        self.specbox_main = QHBoxLayout()
        self.specbox_main.addWidget(self.specplotcanvas)
        self.specbox_main.setAlignment(self.specplotcanvas, QtCore.Qt.AlignVCenter)

        self.specbox_result = QHBoxLayout()
        self.specbox_result.addWidget(self.resultcanvas)
        self.specbox_result.setAlignment(self.resultcanvas, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.specbox_main)
        self.layout.addLayout(self.specbox_result)

        self.setWindowTitle("Interactive SpecOneD GUI - ResultPlotMode")

        self.gcid = self.specplotcanvas.mpl_connect('key_press_event',
                                                    self.on_press_simple)

        self.mpl_toolbar = NavigationToolbar(self.specplotcanvas, self.main_widget)
        self.mpl_toolbar = NavigationToolbar(self.resultcanvas, self.main_widget)

        # NOW INTEGRATE MORE FUNCTIONALITY AND THE RESULT REPLOTTING FUNCTION
        # self.rcid = self.resultcanvas.mpl_connect('key_press_event',
                                                    # self.on_press_simple)


    def SimplePlotMode(self, normalize = True):
        """ This function activates the simple plotting mode

        Parameters
        ----------
        normalize : boolean, optional
            Boolean that decides whether all spectra will be normalized to the
            zeroth spectrum in the sequence to present them on one scale.
        """

        self.mode = "simple"

        self.main_widget = QtWidgets.QWidget(self)

        if normalize:
            for spec in self.in_dict['spec_list'][:self.act] + self.in_dict['spec_list'][self.act+1:]:

                        spec.renormalize_by_spectrum(
                            self.in_dict['spec_list'][self.act],
                            trim_mode='wav',
                            inplace=True)

        # Layout of the main GUI
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        self.in_dict['x_lo'] = min(self.in_dict['spec_list'][self.act].dispersion)
        self.in_dict['x_hi'] = max(self.in_dict['spec_list'][self.act].dispersion)
        self.in_dict['y_lo'] = min(self.in_dict['spec_list'][self.act].flux)
        self.in_dict['y_hi'] = max(self.in_dict['spec_list'][self.act].flux)

        self.tx1 = min(self.in_dict['spec_list'][self.act].dispersion)
        self.tx2 = max(self.in_dict['spec_list'][self.act].dispersion)

        self.specplotcanvas = SpecPlotCanvas(self, in_dict=self.in_dict, active=self.act)

        self.layout.addWidget(self.specplotcanvas)

        self.specplotcanvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.specplotcanvas.setFocus()

        self.setCentralWidget(self.main_widget)

        self.mpl_toolbar = NavigationToolbar(self.specplotcanvas, self.main_widget)

        self.setWindowTitle("Interactive SpecOneD GUI - SimplePlotMode")

        self.gcid = self.specplotcanvas.mpl_connect('key_press_event',
                                                    self.on_press_simple)


    def GetValue(self):

        return self.in_dict['spec_list'][self.act]

    def on_press_simple(self, event):
        """ This function presents the key press options for the general mode.

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key == "1":
            self.act = 0
            self.specplotcanvas.plot(self.in_dict, self.act)

        elif event.key == "2":
            self.act = 1
            self.specplotcanvas.plot(self.in_dict, self.act)

        elif event.key == "3":
            self.act = 2
            self.specplotcanvas.plot(self.in_dict, self.act)

        elif event.key == "e":
            self.specplotcanvas.mpl_disconnect(self.gcid)
            self.statusBar().showMessage("Mode: Zoom", 5000)
            self.handleStart()

        elif event.key == "t":
            self.specplotcanvas.mpl_disconnect(self.gcid)
            self.set_trim_hbox()

            self.statusBar().showMessage("Mode: Trim Dispersion", 5000)
            self.cid = self.specplotcanvas.mpl_connect('key_press_event',
                                                       self.on_press_trim)
        elif event.key == "s":
            self.specplotcanvas.mpl_disconnect(self.gcid)
            self.scale = self.in_dict['spec_list'][self.act].scale
            self.dscale = 0.1
            self.shift = z_to_velocity(self.in_dict['spec_list'][self.act].z)
            self.dshift = 5
            self.yshift = self.in_dict['spec_list'][self.act].yshift
            self.dyshift = 20
            self.in_dict['spec_list'][self.act].unscaled_flux = self.in_dict['spec_list'][self.act].flux

            self.set_scaleshift_hbox()
            self.statusBar().showMessage("Mode: Scale spectrum", 5000)
            self.cid = self.specplotcanvas.mpl_connect('key_press_event',
                                                       self.on_press_scaleshift)

        elif event.key == "v":
            self.specplotcanvas.mpl_disconnect(self.gcid)

            self.shift = z_to_velocity(self.in_dict['spec_list'][self.act].z)
            self.dshift = 5
            self.yshift = self.in_dict['spec_list'][self.act].yshift

            self.set_shift_hbox()
            self.statusBar().showMessage("Mode: Velocity Shift", 5000)
            self.cid = self.specplotcanvas.mpl_connect('key_press_event',
                                                       self.on_press_shift)

        elif event.key == "m":
            self.specplotcanvas.mpl_disconnect(self.gcid)
            self.set_masking_hbox()
            self.statusBar().showMessage("Mode: Interactive Masking", 5000)
            self.cid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_masking)

        elif event.key == "f":
            self.specplotcanvas.mpl_disconnect(self.gcid)
            self.fit_func = "legendre"
            self.order = 6
            self.clip_sig = 2.5
            self.clip_bsize = 120
            self.set_fitting_hbox()
            self.clip_flux = False
            self.statusBar().showMessage("Mode: Interactive Continuum Fitting", 5000)
            self.cid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_fitting)

        elif event.key == "q":
            self.close()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


    def handleStart(self):
        """ Starting an interactive handling loop, which closes once a lower
        and an upper zoom boundary have been interactively defined. It also
        closes on pressing "r" resetting the plotting boundaries.

        """

        self._wx1 = None
        self._wx2 = None
        self._running = True
        self.cid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_set_ranges)
        while self._running:
            QtWidgets.QApplication.processEvents()
            time.sleep(0.05)

            if (self._wx1 is not None and self._wx2 is not None):
                self.in_dict['x_lo'] = self._wx1
                self.in_dict['x_hi'] = self._wx2
                self.in_dict['x2_lo'] = self._wx1
                self.in_dict['x2_hi'] = self._wx2
                self._running = False
                self.statusBar().showMessage("Zommed to %d - %d" % (self._wx1, self._wx2), 5000)
                self.specplotcanvas.mpl_disconnect(self.cid)
                self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
                self.set_plot_ranges()



    def handleStop(self):
        """ Function stopping the interactive loop.

        """

        self._running = False
        self.specplotcanvas.mpl_disconnect(self.cid)
        self.statusBar().showMessage("Stopped Zoom mode", 5000)
        self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)

    def set_plot_ranges(self):
        """ Setting the new plot boundaries by calling all active plots.

        """

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)



    def on_press_set_ranges(self, event):
        """ This function presents the key press options for the mode>
        "Trim dispersion".

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key == "a":
            self._wx1 = event.xdata
            self.statusBar().showMessage("Setting left limit to %d" % (event.xdata), 2000)
        elif event.key == "d":
            self._wx2 = event.xdata
            self.statusBar().showMessage("Setting right limit to %d" % (event.xdata), 2000)
        elif event.key == "q":
            self.handleStop()
        # elif event.key == "s":
        #     self.container[4] = event.ydata
        #     self.statusBar().showMessage("Setting lower limit to %d" % (event.ydata), 2000)
        #     self.set_plot_ranges()
        # elif event.key == "w":
        #     self.container[5] = event.ydata
        #     self.statusBar().showMessage("Setting upper limit to %d" % (event.ydata), 2000)
        #     self.set_plot_ranges()
        elif event.key == "r":
            self.statusBar().showMessage("Resetting plot limits", 2000)

            self.in_dict['x_lo'] = min(self.in_dict['spec_list'][self.act].dispersion)
            self.in_dict['x_hi'] = max(self.in_dict['spec_list'][self.act].dispersion)
            self.in_dict['y_lo'] = min(self.in_dict['spec_list'][self.act].flux)
            self.in_dict['y_hi'] = max(self.in_dict['spec_list'][self.act].flux)

            self.in_dict['x2_lo'] = self.in_dict['x_lo']
            self.in_dict['x2_hi'] = self.in_dict['x_hi']

            self.set_plot_ranges()

            self.handleStop()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def set_trim_hbox(self):
        """ Thus function sets up the hbox layout for the trim dispersion mode
        of the GUI.

        """

        self.trimlo_lbl = QLabel("Lower trim bound:")
        self.trimlo_val = QLabel("{0:.2f}".format(self.tx1))
        self.trimhi_lbl = QLabel("Upper trim bound:")
        self.trimhi_val = QLabel("{0:.2f}".format(self.tx2))

        self.trimlo_in = QLineEdit("{0:.2f}".format(self.tx1))
        self.trimlo_in.setMaxLength(7)

        self.trimhi_in = QLineEdit("{0:.2f}".format(self.tx1))
        self.trimhi_in.setMaxLength(7)

        self.hbox = QHBoxLayout()

        for w in [self.trimlo_lbl, self.trimlo_val, self.trimlo_in, self.trimhi_lbl, self.trimhi_val, self.trimhi_in]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.hbox)


        self.trimlo_in.returnPressed.connect(self.trim_dispersion)
        self.trimhi_in.returnPressed.connect(self.trim_dispersion)


    def on_press_trim(self, event):
        """ This function presents the key press options for the mode>
        "Trim dispersion".

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """


        if event.key == "A":
            self.tx1 = event.xdata
            self.trimlo_val.setText("{0:.2f}".format(self.tx1))
            self.trimlo_in.setText("{0:.2f}".format(self.tx1))
            self.statusBar().showMessage("Setting lower trim bound to %d" % (event.xdata), 2000)

        elif event.key == "D":
            self.tx2 = event.xdata
            self.trimhi_val.setText("{0:.2f}".format(self.tx2))
            self.trimhi_in.setText("{0:.2f}".format(self.tx2))
            self.statusBar().showMessage("Setting upper trim bound to %d" % (event.xdata), 2000)

        elif event.key == "t":
            self.trim_dispersion()

        elif event.key == "T":
            self.trim_all()

        elif event.key == "r":
            self.in_dict['spec_list'][self.act].restore()
            self.specplotcanvas.plot(self.in_dict, self.act)

        elif event.key == "q":
            self.specplotcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.hbox)

            self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
            self.statusBar().showMessage("General Mode", 5000)

    def trim_dispersion(self):
        """ This function trims the dispersion of the spectrum to the specified
        values.

        """

        self.tx1 = float(self.trimlo_in.text())
        self.tx2 = float(self.trimhi_in.text())
        self.trimlo_val.setText("{0:.2f}".format(self.tx1))
        self.trimhi_val.setText("{0:.2f}".format(self.tx2))

        self.statusBar().showMessage("Trimming spectra to %d and %d" % (self.tx1, self.tx2), 2000)

        spec_active = self.in_dict['spec_list'][self.act]
        spec_active.trim_dispersion([self.tx1, self.tx2], inplace=True)
        self.in_dict['spec_list'][self.act] = spec_active

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)

    def trim_all(self):
        """ Trims all spectra in 'spec_list' to the specified dispersion
        values.

        """

        self.tx1 = float(self.trimlo_in.text())
        self.tx2 = float(self.trimhi_in.text())
        self.trimlo_val.setText("{0:.2f}".format(self.tx1))
        self.trimhi_val.setText("{0:.2f}".format(self.tx2))

        self.statusBar().showMessage("Trimming ALL spectra to %d and %d" % (self.tx1, self.tx2), 2000)

        for spec in self.in_dict['spec_list']:
            spec.trim_dispersion([self.tx1, self.tx2], inplace=True)

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def set_scaleshift_hbox(self):
        """ This function sets up the hbox layout for the scale flux mode of
        the GUI

        """
        self.scale_lbl = QLabel("Abritrary scaling:")
        self.scale_val = QLabel("{0:.3f}".format(self.scale))
        self.scale_in = QLineEdit("{0:.2f}".format(self.scale))
        self.scale_in.setMaxLength(7)

        self.dscale_lbl = QLabel("Abritrary scaling increment:")
        self.dscale_val = QLabel("{0:.3f}".format(self.dscale))
        self.dscale_in = QLineEdit("{0:.2f}".format(self.dscale))
        self.dscale_in.setMaxLength(7)

        self.vshift_lbl = QLabel("Velocity shift (km/s):")
        self.vshift_val = QLabel("{0:.3f}".format(self.shift))

        self.dvshift_lbl = QLabel("Velocity shift increment (km/s):")
        self.dvshift_val = QLabel("{0:.3f}".format(self.dshift))
        self.dvshift_in = QLineEdit("{0:.2f}".format(self.dshift))
        self.dvshift_in.setMaxLength(7)

        self.yshift_lbl = QLabel("Arbitrary flux shift (ADU):")
        self.yshift_val = QLabel("{0:.2f}".format(self.yshift))

        self.dyshift_lbl = QLabel("Arbitrary flux shift increment (ADU):")
        self.dyshift_val = QLabel("{0:.2f}".format(self.dyshift))
        self.dyshift_in = QLineEdit("{0:.2f}".format(self.dyshift))
        self.dyshift_in.setMaxLength(7)

        self.hbox = QHBoxLayout()

        for w in [self.scale_lbl,
                  self.scale_val,
                  self.scale_in,
                  self.dscale_lbl,
                  self.dscale_val,
                  self.dscale_in,
                  self.vshift_lbl,
                  self.vshift_val,
                  self.dvshift_lbl,
                  self.dvshift_val,
                  self.dvshift_in,
                  self.yshift_lbl,
                  self.yshift_val,
                  self.dyshift_lbl,
                  self.dyshift_val,
                  self.dyshift_in]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.hbox)

        self.scale_in.returnPressed.connect(self.set_scale)
        self.dscale_in.returnPressed.connect(self.set_dscale)
        self.dvshift_in.returnPressed.connect(self.set_dvshift)
        self.dyshift_in.returnPressed.connect(self.set_dyshift)

    def set_dyshift(self):
        """ This function sets dyshift to the user input value.

        """

        self.dyshift = float(self.dyshift_in.text())
        self.dyshift_val.setText("{0:.2f}".format(self.dyshift))

    def set_scale(self):
        """ This function sets scale to the user input value.

        """

        self.scale = float(self.scale_in.text())
        self.scale_val.setText("{0:.2f}".format(self.scale))

    def set_dscale(self):
        """ This function sets scale to the user input value.

        """

        self.dscale = float(self.dscale_in.text())
        self.dscale_val.setText("{0:.2f}".format(self.dscale))



    def on_press_scaleshift(self, event):
        """ This function presents the key press options for the mode>
        "Shift dispersion".

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """
        if event.key=="a":
            self.shift -= self.dshift
            self.in_dict['spec_list'][self.act].z = velocity_to_z(self.shift)
            self.vshift_val.setText("{0:.2f}".format(self.shift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
            self.shift_redraw()
        elif event.key=="d":
            self.shift += self.dshift
            self.in_dict['spec_list'][self.act].z = velocity_to_z(self.shift)
            self.vshift_val.setText("{0:.2f}".format(self.shift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
            self.shift_redraw()
        elif event.key=="w":
            self.yshift += self.dyshift
            self.yshift_val.setText("{0:.2f}".format(self.yshift))
            self.in_dict['spec_list'][self.act].yshift = self.yshift
            self.shift_redraw()
        elif event.key=="s":
            self.yshift -= self.dyshift
            self.yshift_val.setText("{0:.2f}".format(self.yshift))
            self.in_dict['spec_list'][self.act].yshift = self.yshift
            self.shift_redraw()
        elif event.key == QtCore.Qt.Key_Up or event.key == "8":
            self.scale += self.dscale
            self.scale_val.setText("{0:.2f}".format(self.scale))
            self.in_dict['spec_list'][self.act].scale= self.scale
            self.in_dict['spec_list'][self.act].flux = self.in_dict['spec_list'][self.act].unscaled_flux * self.scale
            self.shift_redraw()
        elif event.key == QtCore.Qt.Key_Down or event.key =="2":
            self.scale -= self.dscale
            self.scale_val.setText("{0:.2f}".format(self.scale))
            self.in_dict['spec_list'][self.act].scale= self.scale
            self.in_dict['spec_list'][self.act].flux = self.in_dict['spec_list'][self.act].unscaled_flux * self.scale
            self.shift_redraw()
        elif event.key == "r":
            self.in_dict['spec_list'][self.act].restore()
            self.in_dict['spec_list'][self.act].reset_mask()
            self.specplotcanvas.plot(self.in_dict, self.act)
            self.scale = 1.0
            self.in_dict['spec_list'][self.act].unscaled_flux  = self.in_dict['spec_list'][self.act].flux
            self.in_dict['spec_list'][self.act].scale= self.scale
            self.shift_redraw()
        elif event.key == "q":
            self.shift_redraw()
            self.specplotcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.hbox)
            self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
            self.in_dict['spec_list'][self.act].scale= 1.0


    def scaleshift_redraw(self):
        """ Redraw function for the "Shift dispersion" mode

        """

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def set_shift_hbox(self):
        """ This function sets up the hbox layout for the shift dispersion mode
        of the GUI.

        """

        self.vshift_lbl = QLabel("Velocity shift (km/s):")
        self.vshift_val = QLabel("{0:.3f}".format(self.shift))
        self.dvshift_lbl = QLabel("Velocity shift increment (km/s):")
        self.dvshift_val = QLabel("{0:.3f}".format(self.dshift))

        self.yshift_lbl = QLabel("Arbitrary flux shift (ADU):")
        self.yshift_val = QLabel("{0:.2f}".format(self.yshift))

        # self.vshift_in = QLineEdit("{0:.2f}".format(self.shift))
        # self.vshift_in.setMaxLength(7)

        self.dvshift_in = QLineEdit("{0:.2f}".format(self.dshift))
        self.dvshift_in.setMaxLength(7)

        self.hbox = QHBoxLayout()

        # for w in [self.vshift_lbl, self.vshift_val, self.vshift_in, self.dvshift_lbl, self.dvshift_val, self.dvshift_in]:
        for w in [self.vshift_lbl, self.vshift_val, self.dvshift_lbl, self.dvshift_val, self.dvshift_in, self.yshift_lbl, self.yshift_val]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.hbox)

        self.dvshift_in.returnPressed.connect(self.set_dvshift)

    def set_dvshift(self):
        """ This function set dvshift to the user input value.

        """

        self.dshift = float(self.dvshift_in.text())
        self.dvshift_val.setText("{0:.2f}".format(self.dshift))

    def on_press_shift(self, event):
        """ This function presents the key press options for the mode>
        "Shift dispersion".

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key=="a":
            self.shift -= self.dshift
            self.in_dict['spec_list'][self.act].z = velocity_to_z(self.shift)
            self.vshift_val.setText("{0:.2f}".format(self.shift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
            self.shift_redraw()
        elif event.key=="d":
            self.shift += self.dshift
            self.in_dict['spec_list'][self.act].z = velocity_to_z(self.shift)
            self.vshift_val.setText("{0:.2f}".format(self.shift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
            self.shift_redraw()
        elif event.key=="+":
            self.dshift +=5
            self.dvshift_val.setText("{0:.2f}".format(self.dshift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
        elif event.key=="-":
            self.dshift -=5
            self.dvshift_val.setText("{0:.2f}".format(self.dshift))
            self.statusBar().showMessage("velocity shift=%d km/s; dshift=%d km/s" % (self.shift, self.dshift), 2000)
        elif event.key=="w":
            self.yshift += 250
            self.yshift_val.setText("{0:.2f}".format(self.yshift))
            self.in_dict['spec_list'][self.act].yshift = self.yshift
            self.shift_redraw()
        elif event.key=="s":
            self.yshift -= 250
            self.yshift_val.setText("{0:.2f}".format(self.yshift))
            self.in_dict['spec_list'][self.act].yshift = self.yshift
            self.shift_redraw()
        elif event.key == "q":
            self.in_dict['spec_list'][self.act].yshift = 0
            self.shift_redraw()
            self.specplotcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.hbox)
            self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
            self.statusBar().showMessage("Spectral velocity shift is : %d km/s -> General Mode" % (self.shift), 5000)

    def shift_redraw(self):
        """ Redraw function for the "Shift dispersion" mode

        """

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def set_masking_hbox(self):
        """ This function sets up the hbox layout for the masking mode
        of the GUI.

        """

        self.masklo_lbl = QLabel("Lower mask bound:")
        self.masklo_val = QLabel("{0:.2f}".format(self.mx1))
        self.maskhi_lbl = QLabel("Upper mask bound:")
        self.maskhi_val = QLabel("{0:.2f}".format(self.mx2))

        self.masklo_in = QLineEdit("{0:.2f}".format(self.mx1))
        self.masklo_in.setMaxLength(7)

        self.maskhi_in = QLineEdit("{0:.2f}".format(self.mx1))
        self.maskhi_in.setMaxLength(7)

        self.hbox = QHBoxLayout()

        for w in [self.masklo_lbl, self.masklo_val, self.masklo_in, self.maskhi_lbl, self.maskhi_val, self.maskhi_in]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.hbox)

        self.masklo_in.returnPressed.connect(self.upd_mask_values)
        self.maskhi_in.returnPressed.connect(self.upd_mask_values)

    def upd_mask_values(self):
        """ This function sets the masking bounds to the user defined
        input values.

        """
        self.mx1 = float(self.masklo_in.text())
        self.mx2 = float(self.maskhi_in.text())
        self.masklo_val.setText("{0:.2f}".format(self.mx1))
        self.maskhi_val.setText("{0:.2f}".format(self.mx2))


    def on_press_masking(self, event):
        """ This function presents the key press options for the masking
        mode.

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key == "A":
            self.mx1 = event.xdata
            self.masklo_val.setText("{0:.2f}".format(self.mx1))
            self.masklo_in.setText("{0:.2f}".format(self.mx1))
            self.statusBar().showMessage("Setting x1 to %d" % (event.xdata), 2000)
        elif event.key == "D":
            self.mx2 = event.xdata
            self.maskhi_val.setText("{0:.2f}".format(self.mx2))
            self.maskhi_in.setText("{0:.2f}".format(self.mx2))
            self.statusBar().showMessage("Setting x2 to %d" % (event.xdata), 2000)
        elif event.key == "m":
            self.mask()
        elif event.key == "f":
            self.mask_all()
        elif event.key == "r":
            self.in_dict['spec_list'][self.act].reset_mask()
            self.specplotcanvas.plot(self.in_dict, self.act)
        elif event.key == "R":
            self.reset_all()
        elif event.key == "q":
            self.specplotcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.hbox)
            self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
            self.statusBar().showMessage("General Mode", 5000)

    def mask(self):
        """ This functions masks the specified region in the active spectrum.

        """
        self.statusBar().showMessage("Masking spectral region between %d and %d" % (self.mx1, self.mx2), 2000)

        mask_between = np.sort(np.array([self.mx1, self.mx2]))

        spec = self.in_dict['spec_list'][self.act]

        lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
        up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))
        self.in_dict['spec_list'][self.act].mask[lo_index:up_index] = False

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)


    def mask_all(self):
        """ This function masks the specified region in all input spectra.

        """
        self.statusBar().showMessage("Masking spectral region between %d and %d" % (self.mx1, self.mx2), 2000)

        mask_between = np.sort(np.array([self.mx1, self.mx2]))

        for spec in self.in_dict['spec_list']:

            lo_index = np.argmin(np.abs(spec.dispersion - mask_between[0]))
            up_index = np.argmin(np.abs(spec.dispersion - mask_between[1]))
            spec.mask[lo_index:up_index] = False

        self.specplotcanvas.plot(self.in_dict, self.act)
        if self.mode == "divide" or self.mode == "multiply":
            self.resultcanvas.result_plot(self.in_dict, self.mode)

    def reset_all(self):
        """ This function resets the masks of all input spectra.

        """

        for spec in self.in_dict['spec_list']:
            spec.reset_mask()

        self.specplotcanvas.plot(self.in_dict, self.act)


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

    def set_fitting_hbox(self):
        """ This function sets up the hbox layout for the fitting mode
        of the GUI.

        """

        self.contfit_lbl = QLabel("Continuum Fit : ")

        self.fitfunc_lbl = QLabel("func ")
        self.fitfunc_val = QLabel(self.fit_func)
        self.order_lbl = QLabel("order ")
        self.order_val = QLabel("{:d}".format(self.order))

        self.fitfunc_in = QLineEdit(self.fit_func)
        self.fitfunc_in.setMaxLength(10)

        self.order_in = QLineEdit("{:d}".format(self.order))
        self.order_in.setMaxLength(3)

        self.clip_sig_lbl =  QLabel("clip_sig ")
        self.clip_sig_val =  QLabel("{0:.2f}".format(self.clip_sig))
        self.clip_sig_in = QLineEdit("{0:.2f}".format(self.clip_sig))
        self.clip_sig_in.setMaxLength(4)


        self.clip_bsize_lbl =  QLabel("clip_bsize ")
        self.clip_bsize_val = QLabel("{:d}".format(self.clip_bsize))
        self.clip_bsize_in = QLineEdit("{:d}".format(self.clip_bsize))
        self.clip_bsize_in.setMaxLength(4)

        self.clip_flux_chbox = QCheckBox("Clip Flux",self)
        self.clip_flux_chbox.stateChanged.connect(self.clickBox)

        self.hbox = QHBoxLayout()

        for w in [self.contfit_lbl,
                  self.fitfunc_lbl,
                  self.fitfunc_val,
                  self.fitfunc_in,
                  self.order_lbl,
                  self.order_val,
                  self.order_in,
                  self.clip_flux_chbox,
                  self.clip_sig_lbl,
                  self.clip_sig_val,
                  self.clip_sig_in,
                  self.clip_bsize_lbl,
                  self.clip_bsize_val,
                  self.clip_bsize_in,
                  ]:
            self.hbox.addWidget(w)
            self.hbox.setAlignment(w, QtCore.Qt.AlignVCenter)

        self.layout.addLayout(self.hbox)

        self.fitfunc_in.returnPressed.connect(self.upd_fitting_values)
        self.order_in.returnPressed.connect(self.upd_fitting_values)
        self.clip_sig_in.returnPressed.connect(self.upd_fitting_values)
        self.clip_bsize_in.returnPressed.connect(self.upd_fitting_values)

    def clickBox(self,state):
        """ This function updates the fit_flux boolean whenever the QCheckBox
        value changes and executes the continuum fit.

        """

        if state == QtCore.Qt.Checked:
            self.clip_flux = True
        else:
            self.clip_flux = False

        self.fit_continuum()

    def upd_fitting_values(self):
        """ This function assingns the user input to the fitting variables.

        """
        self.fitfunc = self.fitfunc_in.text()
        self.order = int(self.order_in.text())
        self.fitfunc_val.setText(self.fitfunc)
        self.order_val.setText("{:d}".format(self.order))
        self.clip_sig = float(self.clip_sig_in.text())
        self.clip_sig_val.setText("{0:2f}".format(self.clip_sig))
        self.clip_bsize = int(self.clip_bsize_in.text())
        self.clip_bsize_val.setText("{:d}".format(self.clip_bsize))

        self.fit_continuum()


    def on_press_fitting(self, event):
        """ This function presents the key press options for the masking
        mode.

        Parameters
        ----------
        event : key_press_event
            Key press event to evaluate
        """

        if event.key == "c":
            self.fit_continuum()
            self.statusBar().showMessage("Fitting Continuum", 2000)
            self.specplotcanvas.plot(self.in_dict, self.act)

        elif event.key == "q":
            self.specplotcanvas.mpl_disconnect(self.cid)
            self.remove_last_layout(self.hbox)
            self.gcid = self.specplotcanvas.mpl_connect('key_press_event', self.on_press_simple)
            self.statusBar().showMessage("General Mode", 5000)

    def fit_continuum(self):
        """ This function fits the specified polynomial to the active spectrum.

        """


        if self.clip_flux:
            spec = self.in_dict['spec_list'][self.act].sigmaclip_flux(low=self.clip_sig, up=self.clip_sig, binsize=self.clip_bsize, niter=5)

        else:
            spec = self.in_dict['spec_list'][self.act].copy()

        spec.fit_polynomial(func=self.fit_func, order=self.order, inplace=True)

        self.in_dict['spec_list'][self.act].fit_flux = spec.fit_flux
        self.in_dict['spec_list'][self.act].fit_dispersion = spec.fit_dispersion
        self.in_dict['spec_list'][self.act].mask = spec.mask

        self.specplotcanvas.plot(self.in_dict, self.act)

    def fit_region(self):
        pass

    def replace_spectrum_with_fit(self):
        pass

    def save_fit_to_file(self):
        pass

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


def determine_v_shift(spec, spec2):

    app = QtWidgets.QApplication(sys.argv)
    form = SpecOneDGui(spec_list=[spec,spec2])
    form.show()

    app.exec_()

    return form.GetValue().z

def low_order_fit(spec):

    app = QtWidgets.QApplication(sys.argv)
    form = SpecOneDGui(spec_list=[spec,])
    form.show()

    app.exec_()

    return form.GetValue()


# def example():
#     spec = SpecOneD()
#     spec2 = SpecOneD()
#
#     spec.plot()
#     spec2.plot()
#
#     spec.read_from_fits('J034151_L1_combined.fits', unit='f_lam')
#     spec2.read_from_fits('HD24000_L1_combined.fits', unit='f_lam')
#     # spec.read_from_fits('HD287515_avg.fits', unit='f_lam')
#     # spec2.read_from_fits('uka0v.fits', unit='f_lam')
#
#
#     app = QtWidgets.QApplication(sys.argv)
#     form = SpecOneDGui(spec_list=[spec,spec2], mode="divide")
#     form.show()
#
#     app.exec_()
#
#     print (form.GetValue().dispersion)


def telluric_correction(science, telluric, telluric_model):

    # simplified version of telluric correction

    # science = target * extinction_target * atmosphere * throughput
    # telluric = star * extinction_star * atmosphere * throughput
    # telluric_model = star

    # 1) take care of galactic extinction
    # 2) target = science / telluric * telluric_model
    # 2a) science / telluric (with scale and shift telluric)
    # 2b) science * telluric_model (apply same velocity shift as for telluric)
    # 3) normalize flux by magnitude
    # ALTERNATIVELY DIVIDE TELLURIC BY TELLURIC_MODEL BEFORE


    # 1) Extinction corrections

    # 2) Mask out absorption lines in telluric spectrum

    # 2) science / telluric
    app = QtWidgets.QApplication(sys.argv)
    form = SpecOneDGui(spec_list=[spec,spec2], mode="divide")
    form.show()
    app.exec_()


    # divide science by scaled/shifted telluric to get rid of telluric features

def example():

    science = SpecOneD()
    telluric = SpecOneD()
    telluric_model = SpecOneD()

    science.read_from_fits('J034151_L1_combined.fits')
    telluric.read_from_fits('HD24000_L1_combined.fits')
    telluric_model.read_from_fits('uka0v.fits')

    app = QtWidgets.QApplication(sys.argv)
    form = SpecOneDGui(spec_list=[science,telluric], mode="divide")
    form.show()
    app.exec_()

# example()

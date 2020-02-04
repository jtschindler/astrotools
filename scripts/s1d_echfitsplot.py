#!/usr/bin/env python3

import os
import argparse
from astropy.io import fits
from astrotools.speconed import speconed as sod

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Quick plotting routine for 1D echelle pypeit spectra using the 
            speconed 
            module. 
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename', type=str,
                        help='Filename of spectrum to plot')

    # parser.add_argument('-f', '--filename', required=True, type=str,
                        # help='Filename of spectrum to plot')

    parser.add_argument('-s', '--smooth', required=False, type=float,
                        help='Number of pixels for a simple boxcar smoothing '
                             'of the spectrum.')
    # parser.add_argument('--save', required=False, type=bool,
    #                     help='Save it as a csv file.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    spec = sod.SpecOneD()


    sod.pypeit_spec1d_plot(args.filename, show_flux_err=True,
                           mask_values=False,
                           ex_value='OPT', show='flux', smooth=args.smooth)

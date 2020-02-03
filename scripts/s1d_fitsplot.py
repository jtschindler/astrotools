#!/usr/bin/env python3

import os
import argparse
from astropy.io import fits
from astrotools.speconed import speconed as sod

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
            Quick plotting routine for 1D spectra using the speconed module. 
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('filename', type=str,
                        help='Filename of spectrum to plot')

    # parser.add_argument('-f', '--filename', required=True, type=str,
                        # help='Filename of spectrum to plot')

    parser.add_argument('-s', '--smooth', required=False, type=float,
                        help='Number of pixels for a simple boxcar smoothing '
                             'of the spectrum.')
    parser.add_argument('--save', required=False, type=bool,
                        help='Save it as a csv file.')

    parser.add_argument('-ymax', '--ymax', required=False, type=float,
                        help='Setting the maximum value for the y-axis.')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arguments()
    spec = sod.SpecOneD()

    hdu = fits.open(args.filename)

    try:
        if len(hdu) > 1:
            if 'OPT_WAVE' in hdu[1].columns.names:
                spec.read_pypeit_fits(filename=args.filename)

                if args.save:
                    filename = os.path.splitext(args.filename)[0] +'.csv'
                    print(filename)
                    spec.save_to_csv(filename,format='linetools')

                if args.smooth is not None:
                    spec.smooth(args.smooth, inplace=True)

                # spec.sigmaclip_flux(low=4, up=4, inplace=True)
                spec.pypeit_plot(show_flux_err=True, ymax=args.ymax)

        elif len(hdu) > 1 or len(hdu) == 1:
            spec.read_from_fits(filename=args.filename)

            if args.save:
                filename = os.path.splitext(args.filename)[0] +'.csv'
                spec.save_to_csv(filename, format='linetools')

            if args.smooth is not None:
                spec.smooth(args.smooth, inplace=True)


            # spec.sigmaclip_flux(low=4, up=4, inplace=True)
            spec.plot(show_flux_err=True, ymax=args.ymax)
    except:
        raise ValueError("Fits type not understood")

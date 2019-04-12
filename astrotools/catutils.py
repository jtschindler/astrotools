

import os
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u

from dustmaps.sfd import SFDQuery

# Catutils - Catalog Utilities
# This is the place for all functions that manipulate/change/enhance/analyse
# catalog
# data from large astronomic photometric catalogs
# e.g. converting fits to hdf5
# match catalogs by ra/dec etc.



# copied from http://docs.astropy.org/en/stable/_modules/astropy/io/fits/column.html
# L: Logical (Boolean)
# B: Unsigned Byte
# I: 16-bit Integer
# J: 32-bit Integer
# K: 64-bit Integer
# E: Single-precision Floating Point
# D: Double-precision Floating Point
# C: Single-precision Complex
# M: Double-precision Complex
# A: Character
fits_to_numpy = {'L': 'i1', 'B': 'u1', 'I': 'i2', 'J': 'i4', 'K': 'i8',
                'E': 'f4',
              'D': 'f8', 'C': 'c8', 'M': 'c16', 'A': 'a'}




def fits_to_hdf(filename):
    """ Convert fits data table to hdf5 data table.

    :param filename:
    :return:
    """
    hdu = fits.open(filename)
    filename = os.path.splitext(filename)
    df = pd.DataFrame()

    format_list = ['D', 'J']

    dtype_dict = {}

    # Cycle through all columns in the fits file
    for idx, column in enumerate(hdu[1].data.columns):

        # Check whether the column is in a multi-column format
        if len(column.format) > 1 and column.format[-1] in format_list:
            n_columns = int(column.format[:-1])

            # unWISE specific solution
            if column.name[:6] == 'unwise' and n_columns == 2:
                passbands = ['w1', 'w2']

                for jdx, passband in enumerate(passbands):
                    new_column_name = column.name + '_' + passband

                    print(new_column_name)

                    df[new_column_name] = hdu[1].data[column.name][:, jdx]

                    numpy_type = fits_to_numpy[column.format[-1]]
                    dtype_dict.update({new_column_name: numpy_type})

            # SOLUTIONS FOR OTHER SURVEYS MAY BE APPENDED HERE

        # else for single columns
        else:
            print(column.name)
            df[column.name] = hdu[1].data[column.name]
            numpy_type = fits_to_numpy[column.format[-1]]
            dtype_dict.update({column.name: numpy_type})

    # update the dtype for the DataFrame
    print(dtype_dict)
    df.astype(dtype_dict, inplace=True)

    df.to_hdf(filename+'.hdf5', 'data')


def match_catalogs(match_radius, primary_cat, p_ra_col, p_dec_col, \
                    secondary_cat, s_ra_col, s_dec_col, s_columns_to_merge, \
                    s_column_prefix, add_columns_prefix=True,  \
                    verbose=False):
    """ Match  two catalogs via sky position and return the best match

    :param match_radius:
    :param primary_cat:
    :param p_ra_col:
    :param p_dec_col:
    :param secondary_cat:
    :param s_ra_col:
    :param s_dec_col:
    :param s_columns_to_merge:
    :param s_column_prefix:
    :param add_columns_prefix:
    :param verbose:
    :return:
    """

    if verbose:
        print("---------------------------------------------------------------")
        print("----------------Matching And Merging Catalogs------------------")
        print("---------------------------------------------------------------")
        print("The Primary catalog has ",primary_cat.shape[1]," columns.")
        print( len(s_columns_to_merge), \
                " columns of the Secondary catalog will be added")


    coo_p = SkyCoord(primary_cat[p_ra_col].values*u.deg, \
            primary_cat[p_dec_col].values*u.deg)

    coo_s = SkyCoord(secondary_cat[s_ra_col].values*u.deg, \
            secondary_cat[s_dec_col].values*u.deg)

    idx_s, distance_s, d3d = coo_p.match_to_catalog_sky(coo_s)

    primary_cat['index_to_match'] = idx_s
    primary_cat[s_column_prefix+'distance'] = distance_s

    primary_cat.loc[primary_cat.query(s_column_prefix+'distance >'+str(match_radius)+' /3600.'\
                    ).index,'index_to_match'] = np.NaN

    primary_cat.loc[:,s_column_prefix+'match'] = False
    primary_cat.loc[primary_cat.query(s_column_prefix+'distance <'+str(match_radius)+' /3600.'\
                    ).index,s_column_prefix+'match'] = True

    primary_cat.loc[primary_cat.query(s_column_prefix+'distance >'+str(match_radius)+' /3600.'\
                    ).index,s_column_prefix+'distance'] = np.NaN



    merged_cat = primary_cat.merge(secondary_cat[s_columns_to_merge], how='left', \
                        left_on='index_to_match', right_index=True)

    merged_cat.drop(['index_to_match'],axis=1,inplace=True)

    if verbose:
        print("Additionally a column was added for the match DISTANCE")
        print("and a True/False columns indicating if the row was matched")
        print("The Merged catalog has ", merged_cat.shape[1], " columns")
        print("The column prefix for the merged columns is ", s_column_prefix)
        print("The added columns are: ")

    for i,column in enumerate(s_columns_to_merge):
        if add_columns_prefix:
            merged_cat.rename(columns={column : s_column_prefix+column}, \
                                inplace=True)
            if verbose:
                print(s_column_prefix+column)
        else:
            if verbose:
                print(column)

    if verbose:
        print(s_column_prefix+'distance')
        print(s_column_prefix+'match')

    if verbose:
        print("The total number of matches found: ", \
                primary_cat.query(s_column_prefix+'match == True').shape[0])
        print("---------------------------------------------------------------")
        print("\n")

    return merged_cat





# TODO This needs to be updated
# def cross_match_ned(match_radius, df, ra_col, dec_col,verbose=False):
#
#     if verbose:
#         print("---------------------------------------------------------------")
#         print("--------------------Match Catalog to NED-----------------------")
#         print("---------------------------------------------------------------")
#
#     df['ned_match'] = False
#     df['ned_name'] = None
#     df['ned_ra'] = 999
#     df['ned_dec'] = 999
#     df['ned_distance'] = 999
#     df['ned_redshift'] = 999
#
#     qso_flags = np.zeros(df.shape[0])
#
#     for i in df.index:
#         ra = df.loc[i,ra_col].values
#         dec = df.loc[i,dec_col].values
#         # print df.loc[i,'wise_designation'],ra, dec
#
#         try:
#             co = coordinates.SkyCoord(ra=ra, dec=dec,unit=(u.deg, u.deg), frame='fk5')
#             result_table = Ned.query_region(co, radius=match_radius /3600. * u.deg, equinox='J2000.0')
#
#
#             for j in range(len(result_table['Type'])):
#                 if result_table['Type'][j] == "QSO" or result_table['Type'][j] == "QGroup":
#
#             	    if df.loc[i,'ned_distance'] >= result_table['Distance (arcmin)'][j]:
#                 	    df.loc[i,'ned_distance'] = result_table['Distance (arcmin)'][j]*60.
#                 	    df.loc[i,'ned_match'] = True
#                 	    df.loc[i,'ned_name'] = result_table['Object Name'][j]
#                 	    df.loc[i,'ned_redshift'] = result_table['Redshift'][j]
#                 	    df.loc[i,'ned_ra'] = result_table['RA(deg)'][j]
#                 	    df.loc[i,'ned_dec'] = result_table['DEC(deg)'][j]
#
#     	        if verbose:
#                     print result_table['Object Name','RA(deg)','DEC(deg)','Type','Redshift','References'][j]
#
#         except RemoteServiceError:
#           continue
#
#     if verbose:
#         print "The total number of matches found: ", \
#                 df.query('ned_match == True').shape[0]
#         print "----------------------------------------------------------------"
#         print "\n"
#
#     return df




def add_extinction_values(df):

    coords = SkyCoord(catalog['ps_ra'].values*u.deg, catalog['ps_dec'].values*u.deg)

    sfd = SFDQuery()

    ebv = sfd(coords)


    catalog['EBV'] = ebv


    return catalog


def build_adjacent_flux_ratios(df, flux_names):
    """ Build flux ratios from adjacent fluxes in DataFrame.

    :param df: DataFrame
    :param flux_names: list of strings
    :return: DataFrame
        Returns a new DataFrame with the calculated flux ratios
    """

    for idx in range(len(flux_names)-1):

        flux_name_a = flux_names[idx]
        flux_name_b = flux_names[idx+1]

        suffix = flux_name_a.split('_')[-1] + flux_name_b.split('_')[-1]

        new_column_name = 'flux_ratio_'+suffix

        df[new_column_name] = df[flux_name_a/flux_name_b]

    return df_new


def match_example(df, verbose=True):
    """

    :param df:
    :param verbose:
    :return:
    """
    dr14q = pd.read_hdf('quasar_catalogs/DR14Q_v4_4.hdf5', 'data')
    columns_to_merge = ['SDSS_NAME', 'RA', 'DEC', 'Z', 'Z_PIPE_ERR', 'PSFMAG_I', \
                        'GAL_EXT_I', 'Z_VI', 'Z_PIPE', 'BI_CIV']
    df = match_catalogs(4.0, df, 'ps_ra', 'ps_dec', \
                                 dr14q, 'RA', 'DEC', columns_to_merge, \
                                 'DR14Q_', ('', '_DR14Q'),
                                 add_columns_prefix=True, \
                                 verbose=verbose)

    return df
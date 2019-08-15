# basic stuff, conversions between units etc.


import numpy as np
from astropy import constants as const

from astropy import coordinates
from astropy.coordinates import SkyCoord
from astropy import units as u

def skymatch_merge_catalogs(match_radius , primary_cat, p_ra_col, p_dec_col, \
                    secondary_cat, s_ra_col, s_dec_col, s_columns_to_merge, \
                    s_column_prefix, suffixes, add_columns_prefix=True,  \
                    verbose=False):

    if verbose:
        print ("----------------------------------------------------------------")
        print ("----------------Matching And Merging Catalogs-------------------")
        print ("----------------------------------------------------------------")
        print ("The Primary catalog has ",primary_cat.shape[1]," columns.")
        print ( len(s_columns_to_merge), \
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
        print ("Additionally a column was added for the matched DISTANCE")
        print ("and a True/False columns indicating if the row was matched")
        print ("The Merged catalog has ", merged_cat.shape[1], " columns")
        print ("The column prefix for the merged columns is ", s_column_prefix)
        print ("The added columns are: ")

    for i,column in enumerate(s_columns_to_merge):
        if add_columns_prefix:
            merged_cat.rename(columns={column : s_column_prefix+column}, \
                                inplace=True)
            if verbose:
                print (s_column_prefix+column)
        else:
            if verbose:
                print (column)

    if verbose:
        print (s_column_prefix+'distance')
        print (s_column_prefix+'match')

    if verbose:
        print ("The total number of matches found: ", \
                primary_cat.query(s_column_prefix+'match == True').shape[0])
        print ("----------------------------------------------------------------")
        print ("\n")

    return merged_cat


#  flux density, power received per unit area per unit frequency, usually per Hz

#  luminosity, power of radiation

#  monochromatic luminosity, power or radiation per Hz, wavelength

#  flux, power of radiation per unit area, f = L/ (4 pi d^2)

#

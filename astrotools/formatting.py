
import numpy as np
import pandas as pd
# import pdfkit
import os

def decdeg2dms(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd*3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return '{0:+03g}:{1:02g}:{2:05.2f}'.format(degrees, minutes, seconds)


def decra2hms(dra):
    minutes, seconds = divmod(dra/15*3600, 60)
    hours, minutes = divmod(minutes, 60)
    return '{0:02g}:{1:02g}:{2:06.3f}'.format(hours, minutes, seconds)


def decdeg2dms_short(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd*3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return '{0:+03.0f}:{1:02.0f}:{2:02.0f}'.format(degrees, minutes, seconds)


def decdeg2dms_nvss(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd*3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return '{0:+03.0f} {1:02.0f} {2:05.2f}'.format(degrees, minutes, seconds)

def decra2hms_nvss(dra):
    minutes, seconds = divmod(dra/15*3600, 60)
    hours, minutes = divmod(minutes, 60)
    return '{0:02g} {1:02g} {2:05.2f}'.format(hours, minutes, seconds)


def hmsra2decdeg(ra_hms, delimiter=':'):

    if isinstance(ra_hms, float) or isinstance(ra_hms, int):

        return convert_hmsra2decdeg(ra_hms, delimiter=delimiter)

    elif isinstance(ra_hms, np.ndarray):

        ra_deg = np.zeros_like(ra_hms)
        for idx, ra in enumerate(ra_hms):

            ra_deg[idx] = convert_hmsra2decdeg(ra, delimiter=delimiter)

        return ra_deg

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(ra_hms)))


def convert_hmsra2decdeg(ra_hms, delimiter=':'):

    if delimiter is None:
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[2:4])
        ra_seconds = float(ra_hms[4:10])
    if delimiter ==':':
        ra_hours = float(ra_hms[0:2])
        ra_minutes = float(ra_hms[3:5])
        ra_seconds = float(ra_hms[6:12])

    # print(ra_hours, ra_minutes, ra_seconds)
    # print((ra_hours + ra_minutes/60. + ra_seconds/3600.) * 15.)

    return (ra_hours + ra_minutes/60. + ra_seconds/3600.) * 15.


def dmsdec2decdeg(dec_dms, delimiter=':'):


    if isinstance(dec_dms, float) or isinstance(dec_dms, int):

        return convert_hmsra2decdeg(dec_dms, delimiter=delimiter)

    elif isinstance(dec_dms, np.ndarray):

        dec_deg = np.zeros_like(dec_dms)
        for idx, dec in enumerate(dec_dms):
            dec_deg[idx] = convert_dmsdec2decdeg(dec, delimiter=delimiter)

        return dec_deg

    else:
        raise TypeError("Input type {} not understood (float, int, "
                        "np.ndarray)".format(type(dec_dms)))

def convert_dmsdec2decdeg(dec_dms,delimiter=':'):

    if delimiter is None:
        dec_degrees = float(dec_dms[0:3])
        dec_minutes = float(dec_dms[3:5])
        dec_seconds = float(dec_dms[5:10])
    if delimiter is not None:
        dec_degrees = float(dec_dms.split(delimiter)[0])
        dec_minutes = float(dec_dms.split(delimiter)[1])
        dec_seconds = float(dec_dms.split(delimiter)[2])

    # print(dec_dms[0])

    if dec_dms[0] == '-':
        is_positive = False
    else:
        is_positive = True

    dec = abs(dec_degrees) + dec_minutes/60. + dec_seconds/3600.

    if is_positive is False:
        dec = -dec

    return dec


def name_to_degrees(name):
    # epoch = name[0]
    ra_hours = float(name[1:3])
    ra_minutes = float(name[3:5])
    ra_seconds = float(name[5:10])

    dec_degrees = float(name[10:13])
    dec_minutes = float(name[13:15])
    dec_seconds = float(name[15:])

    is_positive = dec_degrees >= 0

    ra = (ra_hours + ra_minutes/60. + ra_seconds/3600.) * 15.
    dec = abs(dec_degrees) + dec_minutes/60. + dec_seconds/3600.

    if is_positive is False:
        dec = -dec

    return ra, dec


def decra2hms_short(dra):
    minutes, seconds = divmod(dra/15*3600, 60)
    hours, minutes = divmod(minutes, 60)
    return '{0:02.0f}:{1:02.0f}:{2:04.1f}'.format(hours, minutes, seconds)


def coord_to_shortname(dra, dd):
    # calculate Dec
    is_positive = dd >= 0
    dd = abs(dd)
    dec_minutes, dec_seconds = divmod(dd*3600, 60)
    dec_degrees, dec_minutes = divmod(dec_minutes, 60)
    dec_degrees = dec_degrees if is_positive else -dec_degrees
    # calculate RA
    ra_minutes, ra_seconds = divmod(dra/15*3600, 60)
    ra_hours, ra_minutes = divmod(ra_minutes, 60)
    return '{0:2}{1:02g}{2:02g}{3:+03g}{4:02g}'.format('PS',
                                                       ra_hours,
                                                       ra_minutes,
                                                       dec_degrees,
                                                       dec_minutes)


def coord_to_name(dra,dd, epoch ='J'):
    # calculate Dec
    is_positive = dd >= 0
    dd = abs(dd)
    dec_minutes, dec_seconds = divmod(dd*3600, 60)
    dec_degrees, dec_minutes = divmod(dec_minutes, 60)
    dec_degrees = dec_degrees if is_positive else -dec_degrees
    # calculate RA
    ra_minutes, ra_seconds = divmod(dra/15*3600, 60)
    ra_hours, ra_minutes = divmod(ra_minutes, 60)

    return '{0:1}{1:02g}{2:02g}{3:05.2f} \
           {4:+03g}{5:02g}{6:05.2f}'.format(epoch,
                                             ra_hours,
                                             ra_minutes,
                                             ra_seconds,
                                             dec_degrees,
                                             dec_minutes,
                                             dec_seconds)


def merge_offset_with_cat(offset_cat, target_cat, ident_col_name, ra_col_name,
                          dec_col_name, mag_col_name, distance_col_name):
    """Function that merges a catalog of offset stars with a target catalog.

    Parameters
    ----------
    offset_cat : pandas DataFrame
        The offset star catalog, which includes the already position
        cross-matched offset stars for each target in the target catalog.

    target_cat : pandas dataframe
        The catalog, which includes all the targets for which offset stars
        should be added.

    ident_col_name : str
        Identifier column name that is unique for each object in the target_cat
        (valid for both input DataFrames)

    ra_col_name : str
        Right Ascencion column name (valid for for both input DataFrames)

    dec_col_name : str
        Declination column name (valid for for both input DataFrames)

    mag_col_name : str
        Magnitude column name (valid for for both input DataFrames)

    distance_col_name : str
        Column name for the absolute match distance between the offset star
        position and it's target position (valid only in offset_cat)


    Returns
    -------
    obs_cat : pandas DataFrame
        The merged catalog which include 1 offset star per target
    """

    # Sort the offset catalog by identifier and smallest match distance
    offset_cat.sort_values(by=[ident_col_name, distance_col_name], inplace=True)
    # Drop all duplicate offset stars from the catalog, keep the one closest to
    # the target.
    offset_cat.drop_duplicates(subset=ident_col_name, inplace=True, keep='first')

    # Rename identifiers of offset stars
    for idx in offset_cat.index:
        offset_cat.loc[idx, ident_col_name] = \
                                str(offset_cat.loc[idx, ident_col_name])+'_OFF'

    for idx in target_cat.index:
        target_cat.loc[idx, ident_col_name] = str(target_cat.loc[idx, ident_col_name])

    # Concatenate the two DataFrames
    df = pd.concat([offset_cat[[ident_col_name, ra_col_name, dec_col_name, mag_col_name]],
                   target_cat[[ident_col_name, ra_col_name, dec_col_name, mag_col_name]]],
                   ignore_index=True)

    # Sort all objects in the concatenated DataFrame by RA and DEC
    df.sort_values(by=[ra_col_name, dec_col_name], inplace=True)

    return df


# def download_sdss_finding_charts(df, ident_col_name, ra_col_name, dec_col_name,
#                                  cwd):
#     """Function that merges a catalog of offset stars with a target catalog.
#
#     Parameters
#     ----------
#     df : pandas DataFrame
#         Pandas DataFrame, which includes all targets for which sdss finding
#         charts should be downloaded.
#
#     ident_col_name : str
#         Unique identifier column name for each target
#
#     ra_col_name : str
#         Right Ascencion column name
#
#     dec_col_name : str
#         Declination column name
#
#     cwd : str
#         String that includes the full path to current working directory
#     """
#
#     # SDSS DR13 URL base
#     url_base = 'http://cas.sdss.org/dr13/en/tools/chart/printchart.aspx?ra='
#
#     # Check if directory to store finding charts exists.
#     # If not, create it.
#     if not os.path.exists(cwd+'/SDSS_DR13_finding_charts'):
#         os.makedirs(cwd+'/SDSS_DR13_finding_charts')
#
#     print (cwd+'/SDSS_DR13_finding_charts')
#
#     for idx in df.index :
#         url = url_base + str(df.loc[idx, ra_col_name]) + '&dec=' + \
#               str(df.loc[idx, dec_col_name]) + \
#               '&scale=0.4&width=512&height=512&opt=GL'
#
#         try:
#             pdfkit.from_url(url, cwd+'/SDSS_DR13_finding_charts/'+str(df.loc[idx, ident_col_name])+'.pdf')
#         except:
#             print ("Finding Chart for "+ str(df.loc[idx, ident_col_name]) +" not found")





def mmtformat(df, filename, ident_col_name, ra_col_name, dec_col_name,
              mag_col_name, ra_pm_col_name=None, dec_pm_col_name=None,
              epoch = 'J2000.0'):
    """Function that converts the DataFrame of the targets into the MMT
    catalog format. The function saves the MMT formatted target list as an ascii
    file named [filename].dat.

    Parameters
    ----------
    df : pandas DataFrame
        The catalog of all targets to observe.

    filename : str
        The desired name of the output catalog file

    ident_col_name : str
        Identifier column name that is unique for each object in the target_cat
        (valid for both input DataFrames)

    ra_col_name : str
        Right Ascencion column name (valid for for both input DataFrames)

    dec_col_name : str
        Declination column name (valid for for both input DataFrames)

    mag_col_name : str
        Magnitude column name (valid for for both input DataFrames)

    ra_pm_col_name : str
        Right Ascencion proper motion column name (valid for for both input
        DataFrames)

    dec_pm_col_name : str
        Declination proper motion column name (valid for for both input
        DataFrames)

    epoch : str
        Epoch of coordinate values (default= 'J2000.0')


    Returns
    -------

    """

    ident = df[ident_col_name]
    ra = df[ra_col_name]
    dec = df[dec_col_name]

    if ra_pm_col_name is None:
        ra_pm = '0.0'
    else:
        ra_pm = df[ra_pm_col_name]
    if dec_pm_col_name is None:
        dec_pm = '0.0'
    else:
        dec_pm = df[dec_pm_col_name]

    mag = df[mag_col_name]


    f = open(filename+'_MMT.dat', 'w')

    for i in df.index:

        f.write('{0:16}{1:13}{2:13}{3:4}{4:5}{5:04.1f} 0 {6:9}\n'.format(ident[i],
                                                   decra2hms(ra[i]),
                                                   decdeg2dms(dec[i]),
                                                   ra_pm,
                                                   dec_pm,
                                                   mag[i],
                                                   epoch))
        print ('{0:15}{1:13}{2:13}{3:4}{4:5}{5:05.2f} {6:9}'.format(ident[i],
                                            decra2hms(ra[i]),
                                            decdeg2dms(dec[i]),
                                            ra_pm,
                                            dec_pm,
                                            mag[i],
                                            epoch))

    f.close()


def soarformat(df, filename, ra_col_name, dec_col_name,
              mag_col_name, mag_name, epoch = '2000'):
    """Function that converts the DataFrame of the targets into the SOAR
    telescope catalog format. The function saves the SOAR formatted target list
    as an ascii file named [filename].dat.

    Parameters
    ----------
    df : pandas DataFrame
        The catalog of all targets to observe.

    filename : str
        The desired name of the output catalog file

    ra_col_name : str
        Right Ascencion column name (valid for for both input DataFrames)

    dec_col_name : str
        Declination column name (valid for for both input DataFrames)

    mag_col_name : str
        Magnitude column name (valid for for both input DataFrames)

    mag_name : str
        Name of the passband for the magnitude column name supplied

    epoch : str
        Epoch of coordinate values (default= '2000')


    Returns
    -------

    """

    ra = df[ra_col_name]
    dec = df[dec_col_name]
    mag = df[mag_col_name]

    f = open(filename+'_SOAR.dat', 'w')

    for idx in df.index:
        f.write('['+str(idx)+']'+str(coord_to_shortname(ra[idx], dec[idx]))+' '
                + str(decra2hms(ra[idx]))+' '+str(decdeg2dms(dec[idx]))
                +' ' + epoch + ' ' +str(mag_name)+'='+str(mag[idx])+'\n')
    f.close()

    df.to_csv(filename+'_SOAR.csv',index=True)




def bgreenformat(catalog,catalogfilename):
    df = catalog
    ra = df.sdss_ra
    dec = df.sdss_dec
    nr = df.index
    V = df.psfmag_i
    B_V = '0.000'
    epoch = '2000'

    f = open(catalogfilename+'_bgreen.cat', 'w')
    f.write('nnn  Catalog')
    f.write(' #  Object             RA        Dec   Epoch   V     B-V \n')
    for i in df.index:
        f.write('{0:3d} {1:15}{2:11}\
                 {3:10}{4:5}{5:06.3f}  \
                 {6:7}              \n'.format(i,
                                               coord_to_shortname(ra[i], dec[i]),
                                               decra2hms_short(ra[i]),
                                               decdeg2dms_short(dec[i]),
                                               epoch,
                                               V[i],
                                               B_V))

        print ('{0:3d} {1:15}{2:11}{3:10}\
              {4:5}{5:06.3f}  {6:7}'.format(i,
                                            coord_to_shortname(ra[i], dec[i]),
                                            decra2hms_short(ra[i]),
                                            decdeg2dms_short(dec[i]),
                                            epoch,
                                            V[i],
                                            B_V))
    f.close()
    return


def vattformat(df, filename, ident_col_name, ra_col_name, dec_col_name,
              mag_col_name, prefix='', star_type='f|S|G0',epoch = '2000'):
    """Function that converts the DataFrame of the targets into the VATT
    Xephem catalog format. The function saves the formatted target list as an
    ascii file named [filename].dat.

    Parameters
    ----------
    df : pandas DataFrame
        The catalog of all targets to observe.

    filename : str
        The desired name of the output catalog file

    ident_col_name : str
        Identifier column name that is unique for each object in the target_cat
        (valid for both input DataFrames)

    ra_col_name : str
        Right Ascencion column name (valid for for both input DataFrames)

    dec_col_name : str
        Declination column name (valid for for both input DataFrames)

    mag_col_name : str
        Magnitude column name (valid for for both input DataFrames)

    prefix : str
        Name prefix for the targets in the VATT Xephem catalog format

    star_type : str
        String indicating the "star_type" for the display on the Xephem Sky Map.
        The default is a 0 magnitude G star, that will be displayed as a yellow
        filled circle.

    epoch : str
        Epoch of coordinate values (default = '2000')

    Returns
    -------

    """

    # Creating the object name
    df['target_name'] = "None"
    for idx in df.index:
        df.loc[idx, 'target_name'] = prefix+df.loc[idx,ident_col_name]


    # Creating the ra and dec values
    df['ra'] = 0
    df['dec'] = 0

    for idx in df.index:
        df.loc[idx, 'ra'] = decra2hms(df.loc[idx, ra_col_name])
        df.loc[idx, 'dec'] = decdeg2dms(df.loc[idx, dec_col_name])

    # Creating the star type
    df.loc[:, 'star_type'] = star_type


    # Creating the epoch
    df['epoch'] = epoch


    # Writing the Xephem file
    f = open(filename+'_VATT.dat', 'w')

    for idx in df.index:
        f.write(str(df.loc[idx, 'target_name'])+','
                + str(df.loc[idx, 'star_type'])
                + ','+str(df.loc[idx, 'ra'])
                + ','+str(df.loc[idx, 'dec'])
                + ','+str(df.loc[idx, mag_col_name])
                + ','+str(df.loc[idx, 'epoch'])+'\n')

    f.close()

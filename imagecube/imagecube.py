# Licensed under a 3-clause BSD style license - see LICENSE.rst

# imagecube
# This package accepts FITS images from the user and delivers images that have
# been converted to the same flux units, registered to a common world 
# coordinate system (WCS), convolved to a common resolution, and resampled to a
# common pixel scale requesting the Nyquist sampling rate.
# Each step can be run separately or as a whole.
# The user should provide us with information regarding wavelength, pixel 
# scale extension of the cube, instrument, physical size of the target, and WCS
# header information.

from __future__ import print_function, division

import sys
import getopt
import glob
import math
import os
import warnings
import shutil
import string
import tempfile

from datetime import datetime
from astropy import units as u
from astropy import constants
from astropy.io import fits
from astropy import wcs
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy import log
from astropy.utils.exceptions import AstropyUserWarning
import astropy.utils.console as console
import montage_wrapper as montage

import numpy as np
import scipy
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
from matplotlib import rc

imagecube_fname = 'imagecube.fits'
datacube_fname = 'datacube.fits'

NYQUIST_SAMPLING_RATE = 3.3
"""
Code constant: NYQUIST_SAMPLING_RATE

Some explanation of where this value comes from is needed.

"""

MJY_PER_SR_TO_JY_PER_ARCSEC2 = u.MJy.to(u.Jy)/u.sr.to(u.arcsec**2)
"""
Code constant: MJY_PER_SR_TO_JY_PER_ARCSEC2

Factor for converting Spitzer (MIPS and IRAC)  units from MJy/sr to
Jy/(arcsec^2)

"""

FUV_LAMBDA_CON = 1.40 * 10**(-15)
"""
Code constant: FUV_LAMBDA_CON

Calibration from CPS to Flux in [erg sec-1 cm-2 AA-1], as given in GALEX
for the FUV filter.
http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html

"""

NUV_LAMBDA_CON = 2.06 * 10**(-16)
"""
Code constant: NUV_LAMBDA_CON

calibration from CPS to Flux in [erg sec-1 cm-2 AA-1], as given in GALEX
for the NUV filter.
http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html

"""

FVEGA_J = 1594
"""
Code constant: FVEGA_J

Flux value (in Jy) of Vega for the 2MASS J filter.

"""

FVEGA_H = 1024
"""
Code constant: FVEGA_H

Flux value (in Jy) of Vega for the 2MASS H filter.

"""

FVEGA_KS = 666.7
"""
Code constant: FVEGA_KS

Flux value (in Jy) of Vega for the 2MASS Ks filter.

"""

WAVELENGTH_2MASS_J = 1.2409
"""
Code constant: WAVELENGTH_2MASS_J

Representative wavelength (in micron) for the 2MASS J filter

"""

WAVELENGTH_2MASS_H = 1.6514
"""
Code constant: WAVELENGTH_2MASS_H

Representative wavelength (in micron) for the 2MASS H filter

"""

WAVELENGTH_2MASS_KS = 2.1656
"""
Code constant: WAVELENGTH_2MASS_KS

Representative wavelength (in micron) for the 2MASS Ks filter

"""

JY_CONVERSION = u.Jy.to(u.erg / u.cm**2 / u.s / u.Hz, 1., 
                        equivalencies=u.spectral_density(u.AA, 1500))  ** -1
"""
Code constant: JY_CONVERSION

This is to convert the GALEX flux units given in erg/s/cm^2/Hz to Jy.

"""

S250_BEAM_AREA = 423
"""
Code constant: S250_BEAM_AREA

Beam area (arcsec^2) for SPIRE 250 band.
From SPIRE Observer's Manual v2.4.

"""
S350_BEAM_AREA = 751
"""
Code constant: S250_BEAM_AREA

Beam area (arcsec^2) for SPIRE 350 band.
From SPIRE Observer's Manual v2.4.

"""
S500_BEAM_AREA = 1587
"""
Code constant: S500_BEAM_AREA

Beam area (arcsec^2) for SPIRE 500 band.
From SPIRE Observer's Manual v2.4.

"""

def print_usage():
    """
    Displays usage information in case of a command line error.
    """

    print("""
Usage: """ + sys.argv[0] + """ --dir <directory> --ang_size <angular_size>
[--flux_conv] [--im_reg] [--im_ref <filename>] [--rot_angle <number in degrees>] 
[--im_conv] [--fwhm <fwhm value>] [--kernels <kernel directory>] [--im_regrid] 
[--im_pixsc <number in arcsec>] [--seds] [--cleanup] [--help]  

dir: the path to the directory containing the <input FITS files> to be 
processed. For multi-extension FITS files, currently only the first extension
after the primary one is used.

ang_size: the field of view of the output image cube in arcsec

flux_conv: perform unit conversion to Jy/pixel for all images not already
in these units.
NOTE: If data are not GALEX, 2MASS, MIPS, IRAC, PACS, SPIRE, then the user
should provide flux unit conversion factors to go from the image's native
flux units to Jy/pixel. This information should be recorded in the header
keyword FLUXCONV for each input image.

im_reg: register the input images to the reference image. The user should 
provide the reference image with the im_ref parameter.

im_ref: user-provided reference image to which the other images are registered. 
This image must have a valid world coordinate system. The position angle of
thie image will be used for the final registered images, unless an
angle is explicitly set using --rot_angle.

rot_angle: position angle (+y axis, in degrees West of North) for the registered images.
If omitted, the PA of the reference image is used.

im_conv: perform convolution to a common resolution, using either a Gaussian or
a PSF kernel. For Gaussian kernels, the angular resolution is specified with the fwhm 
parameter. If the PSF kernel is chosen, the user provides the PSF kernels with
the following naming convention:

    <input FITS files>_kernel.fits

For example: an input image named SI1.fits will have a corresponding
kernel file named SI1_kernel.fits

fwhm: the angular resolution in arcsec to which all images will be convolved with im_conv, 
if the Gaussian convolution is chosen, or if not all the input images have a corresponding kernel.

kernels: the name of a directory containing kernel FITS 
images for each of the input images. If all input images do not have a 
corresponding kernel image, then the Gaussian convolution will be performed for
these images.

im_regrid: perform regridding of the convolved images to a common
pixel scale. The pixel scale is defined by the im_pxsc parameter.

im_pixsc: the common pixel scale (in arcsec) used for the regridding
of the images in the im_regrid. It is a good idea the pixel scale and angular
resolution of the images in the regrid step to conform to the Nyquist sampling
rate: angular resolution = """ + `NYQUIST_SAMPLING_RATE` + """ * im_pixsc

seds:  produce the spectral energy distribution on a pixel-by-pixel
basis, on the regridded images.

cleanup: if this parameter is present, then output files from previous 
executions of the script are removed and no processing is done.

help: if this parameter is present, this message will be displayed and no 
processing will be done.

NOTE: the following keywords must be present in all images, along with a 
comment containing the units (where applicable), for optimal image processing:

    BUNIT: the physical units of the array values (i.e. the flux unit).
    FLSCALE: the factor that converts the native flux units (as given
             in the BUNIT keyword) to Jy/pixel. The units of this factor should
             be: (Jy/pixel) / (BUNIT unit). This keyword should be added in the
             case of data other than GALEX (FUV, NUV), 2MASS (J, H, Ks), 
             SPITZER (IRAC, MIPS), HERSCHEL (PACS, SPIRE; photometry)
    INSTRUME: the name of the instrument used
    WAVELNTH: the representative wavelength (in micrometres) of the filter 
              bandpass
Keywords which constitute a valid world coordinate system must also be present.

If any of these keywords are missing, imagecube will attempt to determine them.
The calculated values will be present in the headers of the output images; 
if they are not the desired values, please check the headers
of your input images and try again.
    """)


def parse_command_line(args):
    """
    Parses the command line to obtain parameters.

    """
    # TODO: get rid of global variables!

    global ang_size
    global image_directory
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global main_reference_image
    global fwhm_input
    global kernel_directory
    global im_pixsc
    global rot_angle

##TODO: switch over to argparse
    parse_status = 0
    try:
        opts, args = getopt.getopt(args, "", ["dir=", "ang_size=",
                                   "flux_conv", "im_conv", "im_reg", "im_ref=",
                                   "rot_angle=", "im_conv", "fwhm=", "kernels=", 
                                   "im_pixsc=","im_regrid", "seds", "cleanup", "help"])
    except getopt.GetoptError, exc:
        print(exc.msg)
        print("An error occurred. Check your parameters and try again.")
        parse_status = 2
        return(parse_status)
    for opt, arg in opts:
        if opt in ("--help"):
            print_usage()
            parse_status = 1
            return(parse_status)
        elif opt in ("--ang_size"):
            ang_size = float(arg)
        elif opt in ("--dir"):
            image_directory = arg
            if (not os.path.isdir(image_directory)):
                print("Error: The directory %s cannot be found" % image_directory)
                parse_status = 2
                return(parse_status)
        elif opt in ("--flux_conv"):
            do_conversion = True
        elif opt in ("--im_reg"):
            do_registration = True
        elif opt in ("--rot_angle"):
            rot_angle = float(arg)
        elif opt in ("--im_conv"):
            do_convolution = True
        elif opt in ("--im_regrid"):
            do_resampling = True
        elif opt in ("--seds"):
            do_seds = True
        elif opt in ("--cleanup"):
            do_cleanup = True
        elif opt in ("--im_ref"):
            main_reference_image = arg
        elif opt in ("--fwhm"):
            fwhm_input = float(arg)
        elif opt in ("--kernels"):
            kernel_directory = arg
            if (not os.path.isdir(kernel_directory)):
                print("Error: The directory %s cannot be found: " % kernel_directory)
                parse_status=2
                return
        elif opt in ("--im_pixsc"):
            im_pixsc = float(arg)

    if (main_reference_image != ''):
        try:
            with open(main_reference_image): pass
        except IOError:
            print("The file %s could not be found in the directory %s. Cannot run without reference image, exiting." 
                  % (main_reference_image, image_directory))
            parse_status = 2
    return(parse_status)

def construct_mef(image_directory, logfile_name):
    # Grab all of the .fits and .fit files in the specified directory
    all_files = glob.glob(os.path.join(image_directory,"*.fit*"))
    # no use doing anything if there aren't any files!
    if len(all_files) == 0:
        return(None)

    # create a new header and hdulist
    prihdu = fits.PrimaryHDU()
    prihdr = prihdu.header
    hdulist = fits.HDUList([prihdu])
    # put some information in the header
    prihdr['CREATOR'] = ('IMAGECUBE', 'Software used to create this file') # TODO: add version
    prihdr['DATE'] = (datetime.now().strftime('%Y-%m-%d'), 'File creation date')
    prihdr['LOGFILE'] = (logfile_name, 'imagecube log file') 

    # get images
    for fitsfile in all_files:
        hdu_fits = fits.open(fitsfile)
        img_extens = find_image_planes(hdu_fits) # find all science extensions
        for extens in img_extens:
            extens_name = '%s[%1d]' % (fitsfile,extens)
            header = hdu_fits[extens].header
            # check to see if image has reasonable scale & orientation 
            # TODO: decide whether this is better-placed elsewhere, better done with montage.mOverlaps ?
            pixelscale = get_pixel_scale(header)
            fov = pixelscale * float(header['NAXIS1'])
            log.info("Checking %s: is pixel scale (%.2f\") < ang_size (%.2f\") < FOV (%.2f\") ?"% (extens_name, pixelscale, ang_size,fov))
            if (pixelscale < ang_size < fov): # now check for wavelength keyword
                try:
	            wavelength = header['WAVELNTH'] 
	            header['WAVELNTH'] = (wavelength, 'micron') # add the unit if it's not already there
                    hdulist.append(hdu_fits[extens].copy())
	            hdulist[-1].header['ORIGFILE'] =  (os.path.basename(extens_name), 'Original file name')
                except KeyError:
	            warnings.warn('Image %s has no WAVELNTH keyword, will not be used' % extens_name, AstropyUserWarning)
            else:
	         warnings.warn("Image %s does not meet the above criteria." % extens_name, AstropyUserWarning) 
        hdu_fits.close() # end of loop over all extensions in file
    # end of loop over files
	
    if len(hdulist) > 1: # we have some valid data!
        # TODO: here is the place where we would sort the HDUs by wavelength, if I knew how
         return(hdulist)
    else:
        return(None)

def get_conversion_factor(header):
    """
    Returns the factor that is necessary to convert an image's native "flux 
    units" to Jy/pixel.

    Parameters
    ----------
    header: FITS file header
        The header of the FITS file to be checked.

    Returns
    -------
    conversion_factor: float
        The conversion factor that will convert the image's native "flux
        units" to Jy/pixel.
    """

    # Give a default value that can't possibly be valid; if this is still the
    # value after running through all of the possible cases, then an error has
    # occurred.
    conversion_factor = 0

    try: # figure out what instrument we're dealing with
        instrument = header['INSTRUME']
    except KeyError: # get this if no 'INSTRUME' keyword
        conversion_factor = 0.0
        return(conversion_factor)

    pixelscale = get_pixel_scale(header)

    if (instrument == 'IRAC'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'MIPS'):
        conversion_factor = (MJY_PER_SR_TO_JY_PER_ARCSEC2) * (pixelscale**2)

    elif (instrument == 'GALEX'):
        wavelength = u.um.to(u.angstrom, float(header['WAVELNTH']))
        f_lambda_con = 0
        # I am using a < comparison here to account for the possibility that
        # the given wavelength is not EXACTLY 1520 AA or 2310 AA
        if (wavelength < 2000): 
            f_lambda_con = FUV_LAMBDA_CON
        else:
            f_lambda_con = NUV_LAMBDA_CON
        conversion_factor = (((JY_CONVERSION) * f_lambda_con * wavelength**2) /
                             (constants.c.to('angstrom/s').value))

    elif (instrument == '2MASS'):
        fvega = 0
        if (header['FILTER'] == 'j'):
            fvega = FVEGA_J
        elif (header['FILTER'] == 'h'):
            fvega = FVEGA_H
        elif (header['FILTER'] == 'k'):
            fvega = FVEGA_KS
        conversion_factor = fvega * 10**(-0.4 * header['MAGZP'])

    elif (instrument == 'PACS'):
        # Confirm that the data is already in Jy/pixel by checking the BUNIT 
        # header keyword
        if ('BUNIT' in header):
            if (header['BUNIT'].lower() != 'jy/pixel'):
                log.info("Instrument is PACS, but Jy/pixel is not being used in "
                      + "BUNIT.")
        conversion_factor = 1.0

    elif (instrument == 'SPIRE'):
        wavelength = float(header['WAVELNTH'])
        if (wavelength == 250):
            conversion_factor = (pixelscale**2) / S250_BEAM_AREA
        elif (wavelength == 350):
            conversion_factor = (pixelscale**2) / S350_BEAM_AREA
        elif (wavelength == 500):
            conversion_factor = (pixelscale**2) / S500_BEAM_AREA
    
    return conversion_factor

def convert_image(hdu, args=None):
    """
    Converts an input image's native "flux units" to Jy/pixel
    The converted values are stored in the list of arrays, 
    converted_data, and they are also saved as new FITS images.

    Parameters
    ----------
    hdu: FITS header/data unit for one image

    """
    if ('FLSCALE' in hdu.header):
        conversion_factor = float(hdu.header['FLSCALE'])
    else:
        conversion_factor = get_conversion_factor(hdu.header)
        # if conversion_factor == 0 either we don't know the instrument
        # or we don't have a conversion factor for it.
        if conversion_factor == 0: 
                warnings.warn("No conversion factor for image %s, using 1.0"\
                     % hdu.header['ORIGFILE'], AstropyUserWarning)
                conversion_factor = 1.0

        # Do a Jy/pixel unit conversion
        hdu.data *= conversion_factor
        hdu.header['BUNIT'] = 'Jy/pixel'
        hdu.header['JYPXFACT'] = (conversion_factor, 'Factor to'
            + ' convert original BUNIT into Jy/pixel.')
    return

#modified from aplpy.wcs_util.get_pixel_scales
def get_pixel_scale(header):
    '''
    Compute the pixel scale in arcseconds per pixel from an image WCS
    Assumes WCS is in degrees.

    Parameters
    ----------
    header: FITS header of image


    '''
    w = wcs.WCS(header)

    if w.wcs.has_cd(): # get_cdelt is supposed to work whether header has CDij, PC, or CDELT
        pc = np.matrix(w.wcs.get_pc())
        pix_scale =  math.sqrt(pc[0,0]**2+pc[0,1]**2) * u.deg.to(u.arcsec)
    else: #       but don't think it does
        pix_scale = abs(w.wcs.get_cdelt()[0]) * u.deg.to(u.arcsec)
    return(pix_scale)

def get_pangle(header):
    '''
    Compute the rotation angle, in degrees,  from an image WCS
    Assumes WCS is in degrees.

    Parameters
    ----------
    header: FITS header of image


    '''
    w = wcs.WCS(header)
    pc = w.wcs.get_pc()
    cr2 = math.atan2(pc[0,1],pc[0,0])*u.radian.to(u.deg)    
    return(cr2)

def merge_headers(montage_hfile, orig_header, out_file):
    '''
    Merges an original image header with the WCS info
    in a header file generated by montage.mHdr.
    Puts the results into out_file.


    Parameters
    ----------
    montage_hfile: a text file generated by montage.mHdr, 
    which contains only WCS information
    orig_header: FITS header of image, contains all the other
    stuff we want to keep

    '''
    montage_header = fits.Header.fromtextfile(montage_hfile)
    new_header = orig_header.copy()
    for key in new_header.keys():
        if key in montage_header.keys():
            new_header[key] = montage_header[key] # overwrite the original header WCS
    if 'CD1_1' in new_header.keys(): # if original header has CD matrix instead of CDELTs:
        for cdm in ['CD1_1','CD1_2','CD2_1','CD2_2']: 
            del new_header[cdm] # delete the CD matrix
        for cdp in ['CDELT1','CDELT2','CROTA2']: 
            new_header[cdp] = montage_header[cdp] # insert the CDELTs and CROTA2
    new_header.tofile(out_file,sep='\n',endcard=True,padding=False,clobber=True)
    return

def get_ref_wcs(hdulist, img_name):
    '''
    get WCS parameters from extension in hdulist which matches img_name
    (TODO: make this work properly for multi-extension input)

    Parameters
    ----------
    img_name: name of FITS image file

    '''
    global rot_angle
    ref_found=False

    for hdu in hdulist[1:]:
        if img_name in hdu.header['ORIGFILE']:
            ref_found = True
            lngref_input = hdu.header['CRVAL1']
            latref_input = hdu.header['CRVAL2']
            try:
                rotation_pa = rot_angle # the user-input PA
            except NameError: # user didn't define it
                log.info('Getting position angle from %s' % img_name)
                rotation_pa = get_pangle(hdu.header)
                log.info('Using PA of %.1f degrees' % rotation_pa)
    if not ref_found:
        raise KeyError('No ORIGFILE keyword containing %s' % img_name)
    return(lngref_input, latref_input, rotation_pa)

def find_image_planes(hdulist):
    """
    Reads FITS hdulist to figure out which ones contain science data

    Parameters
    ----------
    hdulist: FITS hdulist

    Outputs
    -------
    img_plns: list of which indices in hdulist correspond to science data

    """
    n_hdu = len(hdulist)
    img_plns = []
    if n_hdu == 1: # if there is only one extension, then use that
        img_plns.append(0)
    else: # loop over all the extensions & try to find the right ones
        for extn in range(1,n_hdu):
            try: # look for 'EXTNAME' keyword, see if it's 'SCI'
                if 'SCI' in hdulist[extn].header['EXTNAME']:
                    img_plns.append(extn)
            except KeyError: # no 'EXTNAME', just assume we want this extension
                img_plns.append(extn)
    return(img_plns)

def register_image(hdu, args):
    """
    Registers image to a reference WCS

    Parameters
    ----------
    hdu: FITS header/data unit for one image

    args: info about common WCS

    """
#    log.info('Processing plane %s' % hdu.header['ORIGFILE'])

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = args['ref_wcs']
    width_and_height = u.arcsec.to(u.deg, args['ang_size'])

    native_pixelscale = get_pixel_scale(hdu.header)

    # make the new header & merge it with old
    artificial_filename = tempfile.mktemp() 
    montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, 
                              width_and_height, artificial_filename, 
                              system='eq', equinox=2000.0, 
                              height=width_and_height, 
                              pix_size=native_pixelscale, rotation=rotation_pa)
    merge_headers(artificial_filename, hdu.header, artificial_filename)
    # reproject using montage
    outhdu = montage.wrappers.reproject_hdu(hdu, header=artificial_filename)  #, exact_size=True)  
    # replace data and header with montage output - problem here in primary vs image HDu?
    hdu.data = outhdu.data
    hdu.header = outhdu.header
    # delete the file with header info
    os.unlink(artificial_filename)
    return

def convolve_image(hdu, args):
    """
    Convolves image with either a Gaussian kernel or
    other FITS kernel

    Parameters
    ----------
    hdu: FITS header/data unit for one image

    args: info about what kind of convolution to do

    """

    # Check if there is a corresponding PSF kernel.
    # If so, then use that to perform the convolution.
    # Otherwise, convolve with a Gaussian kernel.

    # find kernel TODO: make this work for multi-extension input data
    orig_file_base = os.path.splitext(hdu.header['ORIGFILE'])[0]
    kernel_filename = os.path.join(args['kernel_directory'],orig_file_base + "_kernel.fits")
    log.info("Looking for " + kernel_filename)

    if os.path.exists(kernel_filename):
        log.info("Found a kernel; will convolve with it shortly.")
        # reading the kernel
        kernel_hdulist = fits.open(kernel_filename)
        kernel_image = kernel_hdulist[0].data
        kernel_hdulist.close()
        # do the convolution 
        convolved_image = convolve_fft(hdu.data, kernel_image)
        hdu.header['KERNEL'] = (kernel_filename, 'Kernel used in convolution')
    elif args['fwhm_input'] != '': # no kernel but fwhm_input specified
        # construct kernel
        # NOTETOSELF: not completely clear whether Gaussian2DKernel 'width' is sigma or FWHM
        # also, previous version had kernel being 3x3 pixels which seems pretty small!
        native_pixelscale = get_pixel_scale(hdu.header)
        sigma_input = (fwhm_input / 
                           (2* math.sqrt(2*math.log (2) ) * native_pixelscale))
        gaus_kernel_inp = Gaussian2DKernel(width=sigma_input)
        # Do the convolution 
        convolved_image = convolve(hdu.data, gaus_kernel_inp)
        hdu.header['FWHM'] = (fwhm_input, 'FWHM value used in convolution, in pixels')
    else:
        warnings.warn('No kernel found and no FWHM given: no convolution performed on %s'\
                     % hdu.header['ORIGFILE'], AstropyUserWarning)
        return

    # replace data with convolved version
    hdu.data = convolved_image
    return

def resample_image(hdu, args):
    """
    Resamples image to a given pixel grid.

    Parameters
    ----------
    hdu: FITS header/data unit for one image

    args: info about how to do the resampling

    """
    # figure out the geometry of the resampled images
    width_input = args['ang_size'] / args['im_pixsc'] # NOTETOSELF: doesn't look right, value is in *pixels*
    height_input = width_input                         #            shouldn't it be in degrees?

    # get WCS info for the reference image
    lngref_input, latref_input, rotation_pa = args['ref_wcs']

    # make the header for the resampled images (same for all)
    artificial_header = tempfile.mktemp()
    montage.commands.mHdr(`lngref_input` + ' ' + `latref_input`, width_input, 
                          artificial_header, system='eq', 
                          equinox=2000.0, height=height_input, 
                          pix_size=args['im_pixsc'], rotation=rotation_pa)

    # generate header for regridded image
    merge_headers(artificial_header, hdu.header, artificial_header)
    # do the regrid 
    outhdu = montage.wrappers.reproject_hdu(hdu, header=artificial_header)  
    # delete the header file
    os.unlink(artificial_header)
    # replace data and header with montage output
    hdu.data = outhdu.data
    hdu.header = outhdu.header
    return

def create_datacube(hdulist,  img_dir, datacube_name):
    """
    Creates a data cube from the input HDUlist.

    Parameters
    ----------
    hdulist: list of FITS header/data units

    """
    # make new directory for output, if needed
    new_directory = os.path.join(image_directory,"datacube")
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    # put the image data into a list (not sure this is quite the right way to do it, but seems to work)
    resampled_images=[]
    waves = []
    for hdu in hdulist[1:]:
        resampled_images.append(hdu.data)
        waves.append(hdu.header['WAVELNTH'])

    # grab the WCS info from the first input image
    new_wcs_header = wcs.WCS(hdulist[1].header).to_header()

    # copy other info into the primary header from hdulist[0].header
    for k in ['CREATOR','DATE','LOGFILE','BUNIT','REF_IM']:
        if k in hdulist[0].header.keys():
            new_wcs_header[k] = (hdulist[0].header[k],hdulist[0].header.comments[k])

    # now use the header and data to create a new fits file
    prihdu = fits.PrimaryHDU(header=new_wcs_header, data=resampled_images)
    hdulist = fits.HDUList([prihdu])
    # add checksums to header
    hdulist[0].add_datasum(when='Computed by imagecube')
    hdulist[0].add_checksum(when='Computed by imagecube',override_datasum=True)
    # add wavelength info to header
    wavestr = ''
    for w in waves:
        wavestr+= ' %.1f' % w
    hdulist[0].header['WAVELNTH'] = (wavestr, 'Wavelengths in microns of input data') 

    # NOTETOSELF: user-settable output name?
    hdulist.writeto(os.path.join(new_directory,datacube_name),clobber=True)
    return(hdulist)

def output_mef(hdulist, fname):
    for hdu in hdulist:
        hdu.add_datasum(when='Computed by imagecube')
    hdulist[0].add_checksum(when='Computed by imagecube', override_datasum=True)
    hdulist.writeto(fname, clobber=True, output_verify='fix')
    return


def process_images(process_func, hdulist, args, header_add={}):
    if 'HISTORY' in hdulist[0].header:
        for hist_line in hdulist[0].header['HISTORY']:
            if process_func.__name__ in hist_line:
                warnings.warn('Function %s already run on this imagecube' % process_func.__name__, AstropyUserWarning)

    for hdu in hdulist[1:]: # start at 1 b/c 0 is primary header, no image data
        process_func(hdu, args) # error-trap here?

    # add info to primary header and logfile
    hdulist[0].header['HISTORY'] = 'imagecube: %s completed at %s' % (process_func.__name__,\
                                                                          datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    for key in header_add.keys():
        hdulist[0].header[key] = header_add[key]
    log.info('Function %s complete' % process_func.__name__)
    return


def output_seds(cube_hdu):
    """
    Makes pixel-by-pixel SEDs.

    Parameters
    ----------
    cube_hdu: datacube header/data unit

    """
    # make new directory for output, if needed
    new_directory = os.path.join(image_directory,"seds")
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)

    wavelength = cube_hdu.header['WAVELNTH']

    sed_data = []
    #TODO: finish converting this to use cube_hdu
    for i in range(0, num_wavelengths):
        for j in range(len(all_image_data[i])):
            for k in range(len(all_image_data[i][j])):
                sed_data.append((int(j), int(k), wavelengths[i], 
                                all_image_data[i][j][k]))

    # write the SED data to a test file
    # NOTETOSELF: make this optional?
    data = np.copy(sorted(sed_data))
    np.savetxt('test.out', data, fmt='%f,%f,%f,%f', 
               header='x, y, wavelength (um), flux units (Jy/pixel)')
    num_seds = int(len(data) / num_wavelengths)

    with console.ProgressBarOrSpinner(num_seds, "Creating SEDs") as bar:
        for i in range(0, num_seds):

            # change to the desired fonts
            rc('font', family='Times New Roman')
            rc('text', usetex=True)
            # grab the data from the cube
            wavelength_values = data[:,2][i*num_wavelengths:(i+1)*
                                num_wavelengths]
            flux_values = data[:,3][i*num_wavelengths:(i+1)*num_wavelengths]
            # NOTETOSELF: change from 0-index to 1-index
            x_values = data[:,0][i*num_wavelengths:(i+1)*num_wavelengths] # pixel pos
            y_values = data[:,1][i*num_wavelengths:(i+1)*num_wavelengths] # pixel pos
            fig, ax = plt.subplots()
            ax.scatter(wavelength_values,flux_values)
            # axes specific
            ax.set_xlabel(r'Wavelength ($\mu$m)')					
            ax.set_ylabel(r'Flux density (Jy/pixel)')
            rc('axes', labelsize=14, linewidth=2, labelcolor='black')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(min(wavelength_values), max(wavelength_values)) #NOTETOSELF: doesn't quite seem to work
            ax.set_ylim(min(flux_values), max(flux_values))
            fig.savefig(new_directory + '/' + `int(x_values[0])` + '_' + 
                          `int(y_values[0])` + '_sed.eps')
            bar.update(i)
    return

def cleanup_output_files():
    """
    Removes files that have been generated by previous executions of the
    script.
    """

    for im in (imagecube_fname, datacube_fname):
        filepth = os.path.join(image_directory, im)
        if (os.path.exists(filepth)):
            log.info("Removing " + filepth)
            os.unlink(filepth)
    return

#if __name__ == '__main__':
def main(args=None):
    # should probably get rid of global variables
    global ang_size
    global image_directory
    global main_reference_image
    global fwhm_input
    global do_conversion
    global do_registration
    global do_convolution
    global do_resampling
    global do_seds
    global do_cleanup
    global kernel_directory
    global im_pixsc
    global rot_angle
    ang_size = ''
    image_directory = ''
    main_reference_image = ''
    fwhm_input = ''
    do_conversion = False
    do_registration = False
    do_convolution = False
    do_resampling = False
    do_seds = False
    do_cleanup = False
    kernel_directory = ''
    im_pixsc = ''


    # note start time for log
    start_time = datetime.now()

    # parse arguments
    if args !=None:
        arglist = string.split(args)
    else:
        arglist = sys.argv[1:]
    parse_status = parse_command_line(arglist) 
    if parse_status > 0:
        if __name__ == '__main__':
            sys.exit()
        else:
            return

    if (do_cleanup): # cleanup and exit
        cleanup_output_files()
        if __name__ == '__main__':
            sys.exit()
        else:
            return

    # if not just cleaning up, make a log file which records input parameters
    logfile_name = 'imagecube_'+ start_time.strftime('%Y-%m-%d_%H%M%S') + '.log'
    with log.log_to_file(logfile_name):
    	log.info('imagecube started at %s' % start_time.strftime('%Y-%m-%d_%H%M%S'))
    	log.info('imagecube called with arguments %s' % arglist)

        # check to see if we already have an imagecube file in this directory
        if os.path.exists(os.path.join(image_directory,imagecube_fname)): # use the existing file 
            hdulist = fits.open(os.path.join(image_directory,imagecube_fname)) # NB: problematic if some tasks shouldn't be redone
        else: # create a new file
            hdulist = construct_mef(image_directory, logfile_name)
            if hdulist == None: # no files found, so quit
                warnings.warn('No fits files found in directory %s' % image_directory, AstropyUserWarning )
                if __name__ == '__main__':
                    sys.exit()
                else:
                    return
        # grab the reference WCS info 
        
        # now work on the imagecube
        if (do_conversion):
            process_images(convert_image, hdulist, args=None, header_add = {'BUNIT': ('Jy/pixel', 'Units of image data')})
	
        if (do_registration):
            try:
                ref_wcs = get_ref_wcs(hdulist, main_reference_image)
#                log.info('Successfully found reference WCS')
                process_images(register_image, hdulist, args={'ang_size': ang_size, 'ref_wcs': ref_wcs},
                               header_add = {'REF_IM': (main_reference_image,'Reference image for resampling/registration')})
            except KeyError:
                warnings.warn('Can\'t find reference image %s, no registration performed' % main_reference_image, AstropyUserWarning)
	
        if (do_convolution):
            process_images(convolve_image, hdulist, args={'kernel_directory': kernel_directory, 'fwhm_input':fwhm_input})
	
        if (do_resampling):
            try:
                ref_wcs = get_ref_wcs(hdulist, main_reference_image) 
                process_images(resample_image, hdulist, args={'ang_size': ang_size, 'ref_wcs': ref_wcs, 'im_pixsc': im_pixsc},
                               header_add = {'REF_IM': (main_reference_image,'Reference image for resampling/registration')})
                cube_hdulist = create_datacube(hdulist, image_directory, datacube_fname)
            except KeyError:
                warnings.warn('Can\'t find reference image %s, no resampling performed' % main_reference_image, AstropyUserWarning)

        if (do_seds):
            if do_resampling: # use the datacube we just made
                output_seds(cube_hdulist[0])
            elif os.path.exists(os.path.join(image_directory,datacube_fname)): # see if there's an existing datacube and use that 
                cube_hdulist = fits.open(os.path.join(image_directory,datacube_fname)) 
                output_seds(cube_hdulist[0])
            else:
                warnings.warn('No datacube found in directory' % image_directory, AstropyUserWarning)

        # all done processing, so output MEF hdulist
        output_mef(hdulist, imagecube_fname)
        hdulist.close()

        # all done!
        log.info('All tasks completed.')
        if __name__ == '__main__':
	    sys.exit()
        else:
	    return



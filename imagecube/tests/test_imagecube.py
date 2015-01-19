# test script for imagecube
# modified from montage_wrapper/tests/test_wrappers.py
from __future__ import print_function, division

import os
import shutil
import tempfile
import warnings
from hashlib import md5

import numpy as np
from numpy.testing import assert_allclose
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.tests.helper import pytest
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.data import download_file, clear_download_cache

from .. import imagecube

# Values for fake header input
# could randomize these to make a better test?
cdelt_val = 0.0066667 # in degrees/pixel
crpix_val = 50.5 
cr1val_val = 10.5
cr2val_val = -43.0
crota2_val = 128.9

# location of test data - remote download is currently not working
# instead, download these files by hand and put them in imagecube/data/testimgs
test_data_loc = "http://www.canfar.phys.uvic.ca/vospace/nodes/pbarmby/imagecube/"
test_data_files = ['I1_n5128_mosaic.fits','I2_n5128_mosaic.fits','I3_n5128_mosaic.fits','I4_n5128_mosaic.fits','n5128_pbcd_24.fits']

class TestImagecube(object):

    def setup_class(self):

        # make a fake header to test the helper functions which access the header
        w = WCS(naxis=2)

        w.wcs.crpix = [crpix_val, crpix_val]
        w.wcs.cdelt = np.array([-cdelt_val, cdelt_val])
        w.wcs.crval = [cr1val_val, cr2val_val]
        w.wcs.ctype = [b"RA---TAN", b"DEC--TAN"]
        w.wcs.crota = [0, crota2_val]

        self.header = w.to_header()
        self.header['INSTRUME']= 'MIPS'

        # make a temporary directory for the input and output
        self.tmpdir = tempfile.mkdtemp()

        # get the test data and copy it to the temp directory
        if os.path.exists('../data/testimgs'): # copy from ../data/testimgs if that exists 
            shutil.copytree('../data/testimgs',os.path.join(self.tmpdir,'imagecubetest'))
        else: # download and symlink to temp directory: NOT WORKING
            os.makedirs(os.path.join(self.tmpdir,'imagecubetest'))
            for fname in test_data_files:
                tmpname = download_file(test_data_loc+fname)
                linked_name = os.path.join(self.tmpdir,'imagecubetest/'+fname)
                shutil.copy2(tmpname, linked_name)

# end of class definition


# get rid of the temporary files
    def teardown_class(self):
        shutil.rmtree(self.tmpdir)
        clear_download_cache()
        return


# test the helper functions 
    def test_helpers(self):
        pixscal_arcsec = imagecube.get_pixel_scale(self.header)
        assert_allclose(pixscal_arcsec/3600.0,cdelt_val)
        pa = imagecube.get_pangle(self.header)
        assert_allclose(pa,crota2_val)
        conv_fact1 = imagecube.get_conversion_factor(self.header) # should be MJy/sr to Jy/pix for MIPS
        assert_allclose(conv_fact1,u.MJy.to(u.Jy)/u.sr.to(u.arcsec**2) * (pixscal_arcsec**2))


# test the main imagecube script    
    def test_imagecube(self):
        # go where the test data are
        orig_dir = os.getcwd()
        os.chdir(os.path.join(self.tmpdir,'imagecubetest'))
        # run through the whole procedure
        test_argstr = '--flux_conv --im_reg --im_conv --fwhm=8 --im_regrid --im_pixsc=3.0 --ang_size=300 --im_ref n5128_pbcd_24.fits --dir ./'  
        imagecube.main(args=test_argstr)

        # check the output imagecube
        hdulist_ic = fits.open(os.path.join(self.tmpdir,'imagecubetest/imagecube.fits'))
        assert len(hdulist_ic) == 6
        assert hdulist_ic[1].header['DATASUM'] == '1038490815'
        assert hdulist_ic[3].data.shape == (102,102)

        racen, deccen, crota = imagecube.get_ref_wcs(hdulist_ic,'n5128_pbcd_24.fits') 
        assert racen == 201.36652
        assert deccen == -43.06438
        assert crota == 153.2133
        hdulist_ic.close()

        # check the output datacube
        # check that we get the right shape output & number of non-NaN pixels
        hdulist_dc = fits.open(os.path.join(self.tmpdir,'imagecubetest/datacube/datacube.fits'))
        assert hdulist_dc[0].data.shape == (5,102,102)
        valid = hdulist_dc[0].data[~np.isnan(hdulist_dc[0].data)]
        assert len(valid) == 52020

        # test DATASUM against value previously computed
        assert hdulist_dc[0].header['DATASUM'] == '842801625' 
        hdulist_dc.close()
        os.chdir(orig_dir)
        return


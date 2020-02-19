#!/usr/bin/evn python
"""
Module to create simulation data for the Non-linearity calibration pipeline.

Ref: Deriving the nonlinearity correction function, Short version, EUCL-IPN-TN-7-017
"""

from NMLogging import get_logger

from astropy.io import fits
from multiprocessing import Pool, cpu_count
import numpy as np
import numpy.ma as ma
import pandas as pd
import pathlib
import random
import sys


#readout time, how often a frame is generated (on board? or on ground?)
TREAD = 1.4548    #seconds

#read noise (TBD), e-
#TODO should be drawn randomly for each pixel and each exposure, it’s somewhere between 4 and 12 e-
ReadNoise = 8

#pixel gain, assuming it's constant across the detector
Gain = 1   #e-/ADU

#detector IDs in the non-linearity coefficients data
detids = ['H2RG_1_1','H2RG_1_2','H2RG_1_3','H2RG_1_4',
          'H2RG_2_1','H2RG_2_2','H2RG_2_3','H2RG_2_4',
          'H2RG_3_1','H2RG_3_2','H2RG_3_3','H2RG_3_4',
          'H2RG_4_1','H2RG_4_2','H2RG_4_3','H2RG_4_4']

def read_exp_table(exposure_table_file):
    #read in the exposures table
    names = ['n_exposures', 'flux', 'utr']
    df_exp = pd.read_csv(exposure_table_file, names=names, delim_whitespace=True, skiprows=1)
    return df_exp
 
def write_fits(outfile, data):
    primary_hdu = fits.PrimaryHDU()
    hdul =  fits.HDUList([primary_hdu])
    #hdu = fits.ImageHDU(data=data, header=hdr, name=extname)
    hdu = fits.ImageHDU(data=data)
    hdul.append(hdu)
   
    hdul.writeto(outfile)

#def make_sim(flux, nread, iexp, satlevel=65535):
def make_sim(args):
    """
     Generate simulation data per exposure (per exposure, per read/UTR).  
     For another exposure run, just re-run this function to get another
     series of simulation data given (flux, nread).

     flux -- (e-/s), measured flux from the lamp for a given pixel.  ADU/frame???
     nread -- number of reads since the start of the exposure
     iexp -- exposure count for this UTR

     iexp, flux and nread are the numbers in the 1st, 2nd and 3rd columns in Table 2 from the Ref (see the comments at the top).

     satlevel -- saturation level, can be a map

     #flux = 19     #ADU/frame
     #nread = 400
    """
    flux, nread, iexp = args
    satlevel=65535

    #outdir = "sim_data_flux" + str(flux)
    #outdir = "sim_data_flux" + str(flux) + "_nonl"
    outdir = "sim_" + "exposure" + str(iexp) + "_flux" + str(flux)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True) 
    
    for i in range(1, nread+1):
        logger.info("Looping through UTR {}".format(i))
        #outfileprefix = outdir + "/exposure" + str(iexp) + "_utr" + str(i)
        outfileprefix = outdir + "/utr" + str(i)
        refoutfileprefix = outdir + "/utr" + str(i) + "_ref"

        primary_hdu = fits.PrimaryHDU()
        outhdus =  fits.HDUList([primary_hdu])
        outhdus_ref =  fits.HDUList([primary_hdu])

        ahdu = fits.HDUList([primary_hdu])   #for a coefficients
        bhdu = fits.HDUList([primary_hdu])   #for b coefficients

        #loop all 16 detectors
        ndet = 16
        logger.info("Looping through all 16 detectors")
        for idet in range(ndet):
            detid = detids[idet]
            logger.info("Looping through the detector {}, {}".format(idet+1, detid)) 

            flux_actual = np.zeros((2048, 2048))
            #start with the measured flux, constant spatially but varying temporally
            flux_actual[4:2044,4:2044] = flux * TREAD * i  #fill data region with measured flux, (ADU/f) * s = ADU ???
            flux_actual *= Gain   #e-??
            logger.info("Filled with measured flux")

            #calculate nonlinearity from calibration data, varing spatially but not temporally, unit e-??
            flux_nonlinear = make_nonlinear(flux_actual, outfileprefix, ahdu, bhdu, detid) 
            #flux_nonlinear = flux_actual    #no non-linearity
            logger.info("Made non-linearized flux")

            #add pedestal, constant spatially very varying temporally
            pedestal = random.randint(0, 1000)  #set the random pedistal (between 0 - 1000) for this exposure, unit e-??
            flux_nonlinear[:, :] += pedestal    #add the pedestal to all of the data including the reference pixels.
            logger.info("Added pedestal")

            #add noise (e-)
            readnoise_array = np.random.normal(size=[2048,2048]) * ReadNoise  #generate the read noise, the reference pixels should also have noise. 
            flux_nonlinear += readnoise_array      #add the read noise to the data array
            logger.info("Added read noise")

            # set any pixel above the saturation limit to the saturation limit.
            # This can set to 65535 for all pixels for now, but should be able to 
            # take a map since this will vary pixel to pixel.  
            ind = np.where(flux_nonlinear > satlevel)
            flux_nonlinear[ind] = satlevel
            logger.info("Checked saturation level")

            #add this dector to the final simulated data
            hdu = fits.ImageHDU(data=flux_nonlinear, name=detid)
            outhdus.append(hdu)

            #do reference pixel correction
            updownCorr(flux_nonlinear)
            leftrightCorr(flux_nonlinear)

            #add this dector to another output
            hdu1 = fits.ImageHDU(data=flux_nonlinear, name=detid)
            outhdus_ref.append(hdu1)
            

        #write the sim data for this UTR
        outfile = outfileprefix + ".fits"
        #logger.info("Writing sim data to: {}".format(outfile))
        #outhdus.writeto(outfile)

        refoutfile = refoutfileprefix + ".fits"
        logger.info("Writing reference pixel corrected sim data to: {}".format(refoutfile))
        outhdus_ref.writeto(refoutfile)

        #ahdu.writeto(outfileprefix + "_a_coef.fits")
        #bhdu.writeto(outfileprefix + "_b_coef.fits")

def make_nonlinear(flux_actual, outfileprefix, ahdu, bhdu, detid):
    """
     Make flux non-linear.
     
     Per Peter, this time we'll just generate our own co-efficients rather than trying to invert the old ones.

     Here is a formula that I think will work to generate reasonable co-efficents.  This should be done for each pixel.

     I'm simplifying things here for now by setting the minimum value for non-linearity to 0 (min-NL), and setting the 
     zero'th order term to 0 as well.  
     Pick a random value between 60,000 and 65,535 that is the point where the pixel hits the maximum non-linearity.  Lets call this maxNL
     Pick a random number between 0 and 0.05 (0 and 5%) that sets the non-linearity at the value picked above.  Lets call this NLvalue
     Pick a number between 0.5 and 1 that sets the fraction of the non-linearity in the first order term.  Lets call this fracLinear

     The first order linear term a is then: a = 1 - (NLvalue * fracLinear)

     The second order term is then: b = -1.0*(NLvalue * (1-fracLinear))/maxNL

     The output, non-linear value that we put into the simulation is then:
      output =  a * flux_linear + b * flux_linear^2

     where flux_linear is the input flux that increases linearly with time.

     We want to save the above values for each pixel in an array so we can check against them later. 
     
    """
    flux_nonlinear = np.full_like(flux_actual, 0.0)
    a = np.full_like(flux_actual, 0.0)
    b = np.full_like(flux_actual, 0.0)
    dims = flux_actual.shape

    for i in range(dims[0]):
      for j in range(dims[1]):
         maxnl = random.randint(60000, 65535)
         nl_value = random.uniform(0, 0.05)
         frac_linear = random.uniform(0.5, 1.)

         aa = 1. - (nl_value * frac_linear)
         bb = -1.0 * (nl_value * (1. - frac_linear))/maxnl
         a[i,j] = aa
         b[i,j] = bb

         flux_nonlinear[i,j] = aa * flux_actual[i,j] + bb * np.power(flux_actual[i,j], 2)    

    #save to HDU list
    hdu = fits.ImageHDU(data=a, name=detid)
    ahdu.append(hdu)
    hdu = fits.ImageHDU(data=b, name=detid)
    bhdu.append(hdu)

    return flux_nonlinear

def execute(exposure_table_file):
    logger.info("Read the exposure table..")
    exp_tbl = read_exp_table(exposure_table_file)

    #loop Table 2
    for row in exp_tbl.itertuples():
    #if 1 == 1:
        logger.info("Looping through the exposure row {}".format(row))
        nexp = row.n_exposures
        flux = row.flux
        utr  = row.utr

        #row = exp_tbl.iloc[0]
        #flux = 441
        #utr = 100
        #nexp = 1

        #loop 'N exposures' in 1st column of Table 2
        for iexp in range(nexp):
            logger.info("Looping through the exposure {}".format(iexp+1)) 
            make_sim(flux, utr, iexp+1)

def loop_make_sim(exposure_table_file):
    logger.info("Read the exposure table..")
    exp_tbl = read_exp_table(exposure_table_file)

    #prepare data for parallel processing
    data = []
    #loop Table 2
    for row in exp_tbl.itertuples():
        nexp = row.n_exposures
        flux = row.flux
        utr  = row.utr

        #loop 'N exposures' in 1st column of Table 2
        for iexp in range(nexp):
            data.append([flux, utr, iexp+1])


    #start parallel processing
    with Pool(cpu_count()) as p:
         res = p.map(make_sim, data)

def updownCorr(inputarray):
        """
         Up and Down reference pixel correction.
         
         Args:
             inputarray: 2D numpy array, containing the science data layer

         Returns:
             Implicit return.  The input inputarray got modified in this funtion.
        """

        dims = inputarray.shape
        num_ch = 32
        npix = 4
        mpix = 64

        for ch in range(0, num_ch):
            punmasked = []

            #Up reference pixels
            pud = inputarray[0:npix, ch*mpix:(ch+1)*mpix]
            res = getstats(pud)
            punmasked += list(res.data[res.mask])

            #Down reference pixels
            pud = inputarray[dims[0]-npix:dims[0], ch*mpix:(ch+1)*mpix]
            res = getstats(pud)
            punmasked += list(res.data[res.mask])

            x = np.mean(np.array(punmasked))

            #inputarray[npix:dims[0]-npix, ch*mpix:(ch+1)*mpix] -= x  ##Old doc
            inputarray[0:dims[0], ch*mpix:(ch+1)*mpix] -= x

        return

def leftrightCorr(inputarray):
        """
         Left-Right reference pixels correction

         Args:
             inputarray: 2D numpy array, containing the science data layer

         Returns:
             Implicit return.  The input inputarray got modified in this funtion.
        """

        dims = inputarray.shape
        npix = 4
        nn = 4

        #Left reference pixels
        #plr = inputarray[npix:dims[0]-npix, 0:npix] ##Old doc
        plr = inputarray[0:dims[0], 0:npix]

        resl  = getstats(plr)

        #Right reference pixels
        #plr = inputarray[npix:dims[0]-npix, dims[1]-npix:dims[1]] ##Old doc
        plr = inputarray[0:dims[0], dims[1]-npix:dims[1]]

        resr = getstats(plr)

        for i in range(npix, dims[0]-npix):
            #compute the average over a 2*n+1 window of the left and right
            #reference pixels
            punmasked = []
            pll = resl[i-nn:i+nn+1, :]          
            prr = resr[i-nn:i+nn+1, :]          
            punmasked += list(pll.data[pll.mask])
            punmasked += list(prr.data[prr.mask])
            y = np.mean(np.array(punmasked))

            inputarray[i, npix:dims[1]-npix] -= y

        return

def getstats(pixelarray):
       med = np.median(pixelarray)
       nmad = 1.4826 * np.median(np.fabs(pixelarray- med))
       masku = (pixelarray > (med+5*nmad)) | (pixelarray < (med-5*nmad))
       res = ma.array(pixelarray, mask=~masku)
       #A True in the retrun mask means the pixel is NOT masked by the criteria!
       return res

if __name__ == "__main__":

   #log
   logger = get_logger("MakeNonlinearity", 'make_nonlinearity.log')

   args = ""
   for arg in sys.argv:
       args += " " + arg
   logger.info("Running: python " + args)

   exptblfile = sys.argv[1]
   logger.info("Exposure Table 2 file: {}".format(exptblfile))

   logger.info("Making simulated data for non-linearity...")
   #execute(exptblfile) 
   loop_make_sim(exptblfile) 


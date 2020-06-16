from astropy.io import fits
from astropy.io import ascii
from make_random_trace import make_random_trace
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import datetime
import os
import shutil
import json
import re
import sys

plt.style.use('seaborn-white')
sys.stdout.flush()


def MACCsimulator( ng=15, nf=16, nd=11, darkcurrent = 0.04, lightcurrent = 10., 
                   nonlin = -5.0e-5, readnoise = 9., 
                   RTN1 = False, RTN2 = False, 
                   plots = True ):


    # Frame time
    tf = 1.4548 # tf = 1.42 in DPU test Sec. 8.2.

    # FP average e-/ADU gain
    detGain = 1.3 # detGain = 1/1.4 DPU test; typos "8.4"?

    # Dark current e-/sec
    # darkrkcurrent = 0.04 nominal

    # Light current e-/sec
    # lightcurrent = 10.  52 is LED set point. 2 is Zody.

    # Non-linearity 2nd order coefficent
    # nonlin = -5.0e-5 

    # ADU offset added by DPU
    offsetADU = 0. # offsetADU = 1024 or 0 depends on DPU or ground fitting.

    # readnoise in e-.  9-10 e- per frame is nominal.
    sigmaRead = readnoise/detGain # ADU

    # noiseless total fluence
    fluence = ( darkcurrent + lightcurrent )/ detGain  # ADU/sec

    # group and integration times
    tgrp = (nf + nd)*tf
    tint = (ng-1.)*(nf+nd)*tf

    # Readout count
    nreads = int((ng*nf + (ng-1.)*nd))

    # Total exposure time
    texp =  nreads*tf

    # time vector
    read = np.arange(nreads)
    tread = read*tf

    fdark = fluence*tread + nonlin*((fluence*tread)**2) #  curved signal

    ## Sone references from the DPU Manual
    ## P. 48 test
    ##myPixel = [511, 511]
    ##myPixelSlope = 2.179638
    ##myPixelChi2Temp = 30.748461
    ##myPixelSignal = 31.0 

    # Add read noise
    rg = np.random.rand(nreads)            
    fdarkN = fdark + rg * sigmaRead  / np.max(rg)  

    ## Some testing
    ##y[Range(9,12)] += np.float([5,10,7])         # make some peak
    ## define a model: GaussModel + background polynomial
    ##gauss = GaussModel( )                            # Gaussian
    ##gauss += PolynomialModel( 1 )                    # add linear background
    ##gauss.setParameters( np.float([1,1,0.1,0,0]) )    # initial parameter guess
    ##print gauss.getNumberOfParameters()                # 5 (= 3 for Gauss + 2 for line)
    ##gauss.keepFixed( int([2]), np.float([0.1])) 

    # Add RTN.  These should be cast into their own functions at some point.

    ru = np.random.uniform

    # RTN Type 1
    if RTN1:
        # in principle the first two parameters in make_random_trace function, 
        # setting the number of elements and rate, should be set into the main PF  
        # for more user control.  Here the values are just examples, after some 
        # playing around to get them in the RTN ball park. 
        k=0.2
        sRTN, tRTN = make_random_trace(2000,k,0)  
        
        sRTNterp = ru(200)*np.interp(tread,tRTN,sRTN)
        #onoff = np.round(ru( np.float(nreads) ))
        fon = ru(0.0)
        if (fon < 0.5):
            fon = 0.5    #  At least 50%
        for RTN in range(0,nreads):
            # Generate a random number to decide if this readout is telegraphing
            # or not.
            on = ru(0.0)
            if (on <= fon):
                fdarkN[RTN] += sRTNterp[RTN]*np.mean(fdarkN)
        annRTN = "Type 1; fon = "+str(fon)[0:4]

    # RTN Type 2
    if RTN2:
        fon = ru(0.0) # frequency of readouts that should have RTN
        if (fon < 0.2):  
            fon = 0.2    #  At least 20%.
        level = 0.2*np.mean(fdarkN) # RTN maximum offset from nominal
        level = level*ru(0.0)
        for RTN in range(0,nreads):
            # Generate a random number to decide if this readout is telegraphing 
            # or not.
            on = ru(0.0) 
            if (on <= fon):
                fdarkN[RTN] += level
        annRTN = "Type 2; level = "+str(level)[0:5]+" ADU; fon = "+str(fon)[0:4]


    if not (RTN1 or RTN2):
        annRTN = "NONE"

    #  Begin Slope Calculations, following the DPU S/W specification.
    # Set alpha = 0 to remove Poisson noise correlations.
    alpha = (1 - (nf**2))/(3.*nf*(nf+nd))  

    # beta = 2.*detGain*(sigmaRead**2)/(nf*(alpha+1))  # Kubik ChiSq description
    beta = 2.*(sigmaRead**2)/(nf*(alpha+1))   

    # Select the ng*nf frames which have not been dropped by the DPU
    s2 = np.float()
    t2 = np.float()
    for i in range(nreads):
        if i % int(nf+nd) in (range(nf)):
            s = fdarkN[i]
            t = tread[i]
            s2 = np.append(s2, s)
            t2 = np.append(t2, t) 

    # Average each group
    fgrp = np.reshape(s2[1:],[ng,nf])
    tgrp = np.reshape(t2[1:],[ng,nf])

    fgrpavg = np.float()
    tgrpavg = np.float()

    for i in range(ng):
        fga = np.mean(fgrp[i,:])
        tga = np.mean(tgrp[i,:])
        fgrpavg = np.append(fgrpavg, fga)
        tgrpavg = np.append(tgrpavg, tga)

    fgrpavg = fgrpavg[1:]
    tgrpavg = tgrpavg[1:]

    # Sum the signal differences
    # First set which samples to fit, either fgrpavg for the groups, or fdarkN
    # for the full ramp.
    L = fgrpavg   
    # Similarly select the apppropriate time grid, either tgrpavg for group 
    # averages or tf for the full ramp.                                                                                                                                                                                                  
    deltaT = tgrpavg[1]-tgrpavg[0] 
    a = 1
    #  Use ng to fit the group averages, nreads for the full ramp. 
    b = ng

    def sumRange(L,a,b,printslopes):                                                                                                                                                                                                
        s = 0                                                                                                                                                                                                         
        for i in range(a,b):  
            if printslopes:
                if i == a:
                    print('Group slopes:')
                print(i, ' ', L[i] - L[i-1])
            s += pow((L[i] - L[i-1]) + beta,2)  
        return s  
  
    print('sum = ',sumRange(L,a,b,0))    

    # Slope terms
    #slopeDPU_1 = np.sqrt( 1 + 4 * (detGain**2)*( sumRange(L,a,b,1)/((ng - 1)*((1+alpha)**2)) ) )  # Kubik
    slopeDPU_1 = np.sqrt( 1 + 4 * ( sumRange(L,a,b,1)/((ng - 1)*((1+alpha)**2)) ) )  # DPU Manual
    #slopeDPU_0 = (1 + alpha) / (2.*detGain)   # Kubik 
    slopeDPU_0 = (1 + alpha) / 2.  # DPU manual

    slopeDPU = slopeDPU_0 * (slopeDPU_1 - 1) - beta
    print('Slope DPU = ', slopeDPU)

    signalDPU = slopeDPU * (ng - 1) + offsetADU # Downlinked signal in ADU
    print('Signal DPU = ', signalDPU)

    # Quality Factor

    pseudoFlux = np.sqrt ( sumRange(L,a,b,0) / (ng - 1) ) - beta
    print('Pseudo flux ghat_x = ',pseudoFlux)

    QF = ( 2*detGain/(1+alpha) ) * ( (ng - 1)*pseudoFlux - (fgrpavg[ng-1] - fgrpavg[0]) )
    print('Quality Flag = ', QF) 

    # Simple least squares fit to the full ramp for comparison
    coeff = poly.polyfit(tread[1:], fdarkN[1:], 1)
    ffit = poly.polyval(tread[1:], coeff)

    #coeff  = np.polyfit(tread, fdarkN, 1)
    #print(coeff)
    #umodel = np.poly1d(coeff)
    
    chisqr = np.sum((poly.polyval(tread, coeff) - fdarkN) ** 2) / np.std(fdarkN)

    print("Slope Polyfit(1) = ", coeff[0]," Chi-Squared = ", chisqr)

    # Plotting

    if plots:
        figstr = 'MACC('+str(ng)+','+str(nf)+','+str(nd)+')'
        plt.ion()
        fig, ax = plt.subplots(figsize=(8,5.5))

        ax.plot(tread, fdarkN, 'o',linestyle='',
                markersize=5,mfc='none',mec='black',
                label='Dropped Readouts')

        ax.set(xlabel='time (s)', ylabel='signal (ADU)',
               title=figstr)

        ax.tick_params(axis='both', which='both',length=2)

        ax.plot(t2[1:], s2[1:], 'o', linestyle='',
                markersize=5, mfc='blue',mec='none',
                label='Group Readouts')

        ax.plot(tgrpavg,fgrpavg, 'o', linestyle='',
                markersize=7,mfc='red',
                label='Group Averages')
    
        ax.plot(tread[1:], ffit, linestyle=':',color='g',
                label='LSQ')

        plt.axhline(y=signalDPU, color='r', linestyle='--')

        plt.legend(loc='lower right')

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        delY = ymax-ymin
        delX = xmax-xmin

        annX = xmin + 0.05*delX

        annY1 = ymax - 0.2*delY
        sarray = 'fluence: '+np.str(fluence*detGain)+' e-/sec' + '\n' + \
                 'read noise: '+np.str(sigmaRead*detGain)+' e-' + '\n' + \
                 'a2: '+str(nonlin) + '\n' + \
                 'RTN: '+annRTN 
        ax.annotate(sarray,(annX,annY1))

        annY2 = ymax - 0.3*delY
        sarray2 = r'$\alpha$: ' + np.str(alpha)[0:4]+ r' $\beta$: ' +np.str(beta)[0:4] + '\n' + \
                  'DPU Signal: '+np.str(signalDPU)[0:6]+' ADU; QF: '+np.str(QF)[0:4]
        ax.annotate(sarray2,(annX,annY2),color='r')

        plt.show()

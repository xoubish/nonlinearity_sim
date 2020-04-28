from astropy.io import fits
from astropy.io import ascii
import numpy as np
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

def make_random_trace(nelem=2e4, k=1., plots=1):
  ''' make a binary random trace in (-1,1) with exponentially distributed dwell 
      times. nelem = number of jumps. k = rate.
  '''
  k = float(k)
  dt = 1./k
  rg = np.random.rand( np.int(nelem) )
  ra = np.abs(rg)
  ra = ra/np.max(ra)
  # exponentially distributed jumps:
  t = np.log(1/ra)/k
  tr = np.floor(t/dt)
  # print('max trace ',np.max(tr))
  x = []
  s = 3.0e-3
  # do it:
  for j in np.arange(len(tr)):
      for ti in np.arange(np.int(tr[j])):
          x = np.append(x, s*np.ones(ti))
          s = -s
  time = np.arange(len(x))*dt
  spectrum_x = np.abs(np.fft.fft(x))**2
  freq_x = np.fft.fftfreq(len(x), d=dt)
  # freq_x = np.arange(len(x), dtype=float) / (len(x) * dt)
  freq_x = freq_x[:int(len(spectrum_x)/2.)]
  spectrum_x = spectrum_x[:len(freq_x)]
  spectrum_x = spectrum_x/float(len(x)**2 *freq_x[1])
  spectrum_x[1:] = 2*spectrum_x[1:]

  if plots:
    plt.subplot(211)
    plt.plot(time[:], x[:], '-o')
    plt.subplot(212)
    plt.loglog(freq_x, spectrum_x, '.')
    plt.show()

  return x, time



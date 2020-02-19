#!/usr/bin/evn python

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

filelist = sys.argv[1]
print(filelist)

if len(sys.argv) == 3:
   filelist_nl = sys.argv[2]
   print(filelist_nl)

#read in the list of file names
with open(filelist, 'r') as fp:
     files = json.load(fp)
if len(sys.argv) == 3:
   with open(filelist_nl, 'r') as fp:
        files_nl = json.load(fp)

#pixels to plot
#pixels = [[500, 355], [1820,2000], [104, 597], [3,3]]
pixels = [[500, 355], [1820,2000], [3,3]]

#read in data at the pixel
data = np.zeros((len(files), len(pixels)))
i = 0
for file in files:
    ff = fits.open(file)
    j = 0
    for pixel in pixels:
        data[i,j] = ff[1].data[pixel[0], pixel[1]]
        j += 1
    i += 1
    ff.close()

if len(sys.argv) == 3:
   data_nl = np.zeros((len(files_nl), len(pixels)))
   i = 0
   for file in files_nl:
    ff = fits.open(file)
    j = 0
    for pixel in pixels:
        data_nl[i,j] = ff[1].data[pixel[0], pixel[1]]
        j += 1
    i += 1
    ff.close()

#plot the data
m = 3  #number of rows for the plot
n = 2  #number of columns for the plot
fig, axes = plt.subplots(m, n, sharex='col')

for ip,p in enumerate(pixels):
    print(ip)
    print(p)
    #i = int(ip/n)
    #j = ip % m 
    i = ip
    axes[i,0].plot(data[:, ip], label="no nonlinearity")
    axes[i,0].set_title("pixel [" + str(p[0]) + ", " + str(p[1]) + "]")
    axes[i,0].set_xlim(0, len(files))
    
    if len(sys.argv) == 3:
       axes[i,0].plot(data_nl[:, ip], c='r', label="with nonlinearity")

       #axes[i,j].plot(data_nl[:, ip], c='b')

       ts = np.divide(data_nl[:, ip], data[:, ip])
       axes[i,1].plot(ts, label="ratio")
       axes[i,1].legend()

    axes[i,0].legend()

plt.show()


"""
This class represents MACC - Multiple Accumulated sampling mode.
ng - number of equally spaced grops
nf - number of frames per group
nd - number of dropped frames between two succesive groups
"""
#readout time, how often a frame is generated (on board? or on ground?)
TREAD = 1.4548    #seconds

class MACC(object):
   def __init__(self, ng, nf, nd):
       self._ng = ng
       self._nf = nf
       self._nd = nd

   def get_ng(self):
       return self._ng

   def get_nf(self):
       return self._nf

   def get_nd(self):
       return self._nd

   def get_tint_texp(self):
       #
       tf = 1   #TODO does tf == TREAD or 1??  4.1.1
       tint = (self._ng - 1) * (self._nf + self._nd) * tf
       texp = (self._ng * self._nf + (self._ng - 1) * self._nd) * tf
       return tint, texp


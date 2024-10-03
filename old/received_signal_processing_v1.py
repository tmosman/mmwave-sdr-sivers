
# -*- coding: utf-8 -*-
"""

Created on Wed Nov 23 11:41:58 2022
Updated on Wed Mar 29 11:10:42 2023
@author: Tawfik Osman

"""

import time
import numpy as np
import socket
import sys
import os
import scipy.io as sc
import ast
from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy.matlib

from dependencies import Estimator, inspect_IQ_samples


if __name__ == "__main__":
    

    ## Create the figure
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6),dpi=100)
    fig.tight_layout(pad=5.0)
    
    
    
    #file_path = './data/Data_collected_1716953397.8812292/iq_samples/iq_31F3CC3_0_1716953411.7583902_1716953463.7562268.npy'
    file_path ='./iq_31F3CC3_3_1718344905.568729_1718344921.1274981.npy'
    #receive_IQ_data= np.fromfile(file_path,dtype=np.complex64)[2000000:]
    receive_IQ_data = np.load(file_path)[15,6,:][200:]
    print(receive_IQ_data.shape)
    
    Fs = 20e6
    numberOFDMSymbols,modOrder= 10,4
    capture_ack,lts_corr = inspect_IQ_samples(fig,ax,receive_IQ_data,numberOFDMSymbols,modOrder,Fs)
    #plt.plot(lts_corr)
    #plt.show()
    '''
    '''

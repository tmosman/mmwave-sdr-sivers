# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 03:18:00 2024

@author: Tawfik Osman

"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sc
import numpy.matlib
import math
np.random.seed(10)

class configVariables:
    def __init__(self,No_SC,NoOFDMSymbols,ModOrder):
        self.NoOFDMSymbols = NoOFDMSymbols  
        self.ModOrder = ModOrder  # Modulation order (2/4/16 = BSPK/QPSK/16-QAM)
        self.No_SC = No_SC     # Number of subcarries
        if self.No_SC == 64:
            self.cp_len = 16      # Cyclic prefix length
            self.pilot_symbols = np.array([1,1, -1, 1])  
            self.subcarrier_ind_pilot = np.array([7,21,43,57])  # Pilot subcarrier indices
            subcarriers_indices = np.arange(0,64) 
            dc_subcarrier = np.array([0])
            guard_band = np.arange(27,38)
            overhead_subcarriers = np.concatenate((dc_subcarrier,self.subcarrier_ind_pilot,guard_band))
            self.subcarrier_ind_data  = np.delete(subcarriers_indices,overhead_subcarriers)   # Data subcarrier indices

        else:
            print(f'Number of Subcarriers: {self.No_SC} Not supported')

        self.no_data_symbols = self.NoOFDMSymbols * len(self.subcarrier_ind_data)  # Number of data symbols

class signal_generator(configVariables):
    def __init__(self,No_SC,NoOFDMSymbols,ModOrder,factor):
        super().__init__(No_SC,NoOFDMSymbols,ModOrder)
        if int(No_SC) == 64:
            self.lts_time,self.preamble = self.preamble_time_domain_64subcs()
        else:
            print(f'Preamble not desgined at {No_SC}')

        self.Interpolation_rate = factor
        self.tx_scale = 1
    
    
    def preamble_time_domain_64subcs(self):
        ## About STS
        # Subcarrier Index:  -26, -22, -20, -18, -14, -10,  -6,  -2,   2,   6,  10,  14,  18,  20,  22,  26
        # STS Value (BPSK):  +1,  +1,  -1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1,  -1,  +1,  +1


        # STS
        sts_f = np.zeros((1,self.No_SC ),dtype='complex')
        sts_f[0,0:27] = [0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,
                         1+1j,0,0,0,1+1j,0,0]
        sts_f[0,38:]= [0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,
                       0,0,0,1+1j,0,0,0]
        sts_t = np.fft.ifft(np.sqrt(13/6)*sts_f[0,:],64)
        sts_t = sts_t[0:16]
        
        ## About LTS
        #  Subcarrier Index:  -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26
        # LTS Value (BPSK):  +1,  +1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1,  -1,  -1,  +1,  +1,  +1,  +1
        # Guard Bands: Subcarriers at indices [-32, -31, -30, -29, -28, -27, 27, 28, 29, 30, 31, 32] are set to zero.
        # DC Subcarrier: The subcarrier at index 0 is set to zero.

        # LTS 
        lts_t = np.zeros((1,self.No_SC ),dtype='complex')
        lts_f = [0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,
                 1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,
                 -1,-1,1,1,-1,1,-1,1,1,1,1]
        lts_t[0,:] = np.fft.ifft(lts_f,64)
        AGC = np.matlib.repmat(sts_t, 1, 30)
        preamble = np.concatenate((AGC[0,:],lts_t[0,32:],lts_t[0,:],lts_t[0,:]),axis=0)
            
        return lts_t,preamble
    

    def mod_util(self,x,y):
        if y == 0:
         result = x
        elif x==y:
            result = 0
        elif x!=y and y!=0:
            result =  x - math.floor(x/y)*y
        return result

    def bitshift_util(self, a,k):
        if k < 0:
            result = math.floor(a >> abs(k))
        else:
            result = a << k
        return result
    
    def upsample2x(self,data,factor):
        """
        Up sample the received IQ samples by a factor of 2
        
        """
        # Read the captured IQ sample file, filter
        filter_coeff = sc.loadmat('./files_req/filter_array.mat')['interp_filt2']
        
       # data_conv = np.convolve(data_recv[0,:],h[0,:],'full').reshape(1,-1)
        filter_data = sig.upfirdn(filter_coeff[0,:],data[0,:],up=factor).reshape(1,-1)
        
        return filter_data
    
    
    def generate_data_OFDM_symbol(self, no_zeros_to_pad):
        ## Generate a payload of random integers
        tx_data = np.random.randint(0,self.ModOrder, self.no_data_symbols)
        #print(tx_data)

        # Normalized constellation symbols
        modvec_bpsk   =  (1/np.sqrt(2))  *  np.array([-1,1])
        modvec_16qam  =  (1/np.sqrt(10)) * np.array( [-3, -1, +3, +1])
        modvec_64qam  =  (1/np.sqrt(43)) *  np.array([-7 ,-5, -1, -3 ,+7 ,+5, +1 ,+3]) 
        
        if self.ModOrder == 2:
            mod_fcn  = lambda x: complex(modvec_bpsk[x],0)
        elif self.ModOrder == 4:
            mod_fcn  = lambda x: complex(modvec_bpsk[self.bitshift_util(x, -1)], modvec_bpsk[self.mod_util(x, 2)])
        elif self.ModOrder == 16:
            mod_fcn  = lambda x: complex(modvec_16qam[self.bitshift_util(x, -2)], modvec_16qam[self.mod_util(x, 4)])
        elif self.ModOrder == 64:
            mod_fcn  = lambda x: complex(modvec_64qam[self.bitshift_util(x, -3)], modvec_64qam[self.mod_util(x, 8)])
            
        data_out = []
        for data_sample in tx_data:
            data_out.append(mod_fcn(data_sample))
        
        tx_syms = np.array(data_out)
        
        
        # Reshape symbols
        tx_syms_mat = np.matlib.reshape(tx_syms, (self.NoOFDMSymbols,len(self.subcarrier_ind_data)))
        tx_syms_mat = np.transpose(tx_syms_mat)
        pilots_mat = np.transpose(np.vstack([self.pilot_symbols] * self.NoOFDMSymbols))
        
        # Assign symbols to subcarriers
        ifft_in_mat = np.zeros((self.No_SC, self.NoOFDMSymbols),dtype = np.complex64)
        ifft_in_mat[self.subcarrier_ind_data, :]   = tx_syms_mat
        ifft_in_mat[self.subcarrier_ind_pilot, :] = pilots_mat
        
        # Create buffer for the OFDM symbol
        tx_payload_mat = np.zeros((self.No_SC, self.NoOFDMSymbols),dtype = np.complex64)
      
        # Freq. domain to time domain
        tx_payload_mat = np.fft.ifft(ifft_in_mat, self.No_SC, axis=0);
      
        
        # Insert the cyclic prefix
        if self.cp_len > 0: 
            tx_cp = tx_payload_mat[tx_payload_mat.shape[0]-self.cp_len:tx_payload_mat.shape[0],:] 
            tx_payload_mat = np.vstack((tx_cp,tx_payload_mat))
        
        tx_payload_vec = np.matlib.reshape(tx_payload_mat.transpose(), (1,(np.prod(tx_payload_mat.shape))))
        #print(tx_payload_vec.shape)
        # Construct the full time-domain OFDM waveform
        tx_waveform = np.hstack((self.preamble.reshape(1,-1),tx_payload_vec))
        
        # Zero-Padding
        tmp_waveform = np.hstack((tx_waveform,np.zeros((1,int(no_zeros_to_pad/2)))))
        padded_waveform = np.hstack((np.zeros((1,int(no_zeros_to_pad/2))),tmp_waveform))#/max(z1)
        
        # Upsampling
        if self.Interpolation_rate == 1:
            tx_vec_air = padded_waveform
            #tx_vec_air = self.upsample2x(padded_waveform,self.Interpolation_rate)
            
        elif self.Interpolation_rate == 2:
            tx_vec_air = self.upsample2x(padded_waveform,self.Interpolation_rate)
        
        # Scale Waveform
        tx_vec_air_scaled = self.tx_scale * (tx_vec_air / np.max(np.abs(tx_vec_air)))
        #return tx_syms_mat,pilots_mat,tx_payload_mat,padded_waveform,tx_vec_air_scaled
        return tx_vec_air_scaled,tx_payload_vec,padded_waveform
    
class symbol_detector(configVariables):
    def __init__(self,No_SC,NoOFDMSymbols,ModOrder):
        super().__init__(No_SC,NoOFDMSymbols,ModOrder)
        self.sig_gen_obj = signal_generator(No_SC, NoOFDMSymbols, ModOrder,factor =2)
        self.LTS_CORR_THRESH = 0.8
        self.FFT_OFFSET = 4

        pass
    def out(self):
        return self.cp_len 
    
    def decimateSamples(self,data,factor):
        if factor > 1:
            # Read the captured IQ sample file, filter
            filter_coeff = sc.loadmat('./files_req/filter_array.mat')['interp_filt2']
            
            #filter_data = np.convolve(data_recv[0,:],h[0,:],'full').reshape(1,-1)
            filter_data = sig.upfirdn(filter_coeff[0,:],data[0,:],down=factor).reshape(1,-1) 
        else:
            filter_data = data

        
        return filter_data
    

    def detectPeaks(self,data):
        """
        Time synchronization; Detect LTS peaks
        Input:
            data - time domain received IQ samples
        Output:
            lts_corr - autocorrelation output
            numberofPackets - Number of detected packets in the raw samples
            valid_peak_indices - Indices of the detected packets
        """
        self.lts_time, _ = self.sig_gen_obj.preamble_time_domain_64subcs()
        flip_lts = np.fliplr(self.lts_time)
        lts_corr = abs(np.convolve(np.conj(flip_lts[0,:]),data[0,:], mode='full'))
        lts_peaks = np.argwhere(lts_corr > self.LTS_CORR_THRESH*np.max(lts_corr));
        
        #print(len(lts_peaks))
        [LTS1, LTS2] = np.meshgrid(lts_peaks,lts_peaks);
        [lts_second_peak_index,_] = np.where(LTS2-LTS1 == np.shape(self.lts_time)[1]);
        valid_peak_indices = lts_peaks[lts_second_peak_index].flatten()
        #print(lts_second_peak_index)
        
        if len(valid_peak_indices) >=1:
            select_peak = np.argmax(lts_second_peak_index)-1
            payload_ind = valid_peak_indices[select_peak]
            lts_ind = payload_ind-160;
            lts_ind = lts_ind+1
            print('Number of Packets Detected: ',len(valid_peak_indices),
                  f' at {payload_ind}')
            #print(lts_peaks[lts_second_peak_index].flatten())
        else:
            payload_ind,lts_ind = 0,0
            print('No Packet Detected !!!')
        
       
        return lts_corr,len(valid_peak_indices),valid_peak_indices
   
   
    def estimate_carrier_freq_offset(self,dataSamples, payload_ind , do_cfo):
        """
        Estimate and correct CFO
        Input:
            dataSamples - time domain received signal before CFo correction
            lts_index - starting index of the lts samples
            do_cfo - Boolean argument True/False
            
        Output:
            cfo_samples_t - data samples after cfo correction;
        """
        lts_index = (payload_ind-160) + 1
        ## CFO Correction
        if do_cfo == True:
            rx_lts = dataSamples[0,lts_index : lts_index+160] #Extract LTS (not yet CFO corrected)
            rx_lts1 = rx_lts[-64+-self.FFT_OFFSET + 96:-64+-self.FFT_OFFSET +160];  # Check indexing
            rx_lts2 = rx_lts[-self.FFT_OFFSET+96:-self.FFT_OFFSET+160];
            
            #Calculate coarse CFO est
            rx_cfo_est_lts = np.mean(np.unwrap(np.angle(rx_lts2 * np.conj(rx_lts1))));
            rx_cfo_est_lts = rx_cfo_est_lts/(2*np.pi*64);
        else:
            rx_cfo_est_lts = 0;
        
        # Apply CFO correction to raw Rx waveform
        time_vec = np.arange(0,np.shape(dataSamples)[1],1);
        rx_cfo_corr_t = np.exp(-1j*2*np.pi*rx_cfo_est_lts*time_vec);
        cfo_samples_t  = dataSamples  * rx_cfo_corr_t
        
        return cfo_samples_t,rx_cfo_est_lts
    
    def estimate_channel(self,dataSamples,payload_ind):
        """
        compute channel estimates from LTS sequences
        Input:
            dataSamples - time domain received signal after CFo correction
            lts_index - starting index of the lts samples
            
        Output:
            rx_H_est - vector of complex channel gains
        """
        lts_index = (payload_ind-160) + 1
        rx_lts = dataSamples[0,lts_index : lts_index+160]
        rx_lts1 = rx_lts[-64+-self.FFT_OFFSET + 96:-64+-self.FFT_OFFSET +160];  # Check indexing
        rx_lts2 = rx_lts[-self.FFT_OFFSET+96:-self.FFT_OFFSET+160]
        #print(rx_lts1.shape,self.N_SC)
        if rx_lts1.shape[0] and rx_lts2.shape[0] == self.No_SC:
            rx_lts1_f, rx_lts2_f = np.fft.fft(rx_lts1), np.fft.fft(rx_lts2)
            # Calculate channel estimate from average of 2 training symbols
            rx_H_est = np.fft.fft(self.lts_time,64) * (rx_lts1_f + rx_lts2_f)/2
            
        else:
            rx_H_est = np.ones((1,64), dtype=np.complex64)

        return rx_H_est
    
    def equalize_symbols(self,samples,payload_index,H_est):
        """
        Equalizing the frequency domain received signal with channel estimates
        Input:
            samples - time domain received signal after CFo correction
            payload_index - starting index of the payload samples
            channelEstimates - vector of complex channel gains
        Output:
            rx_syms - Equalized, frequency domain received data symbols
        """
        rx_dec_cfo_corr = samples[0,:]
        print(f'Received data shape : {rx_dec_cfo_corr.shape}')
        payload_vec = rx_dec_cfo_corr[payload_index+1: payload_index+self.NoOFDMSymbols*(self.No_SC+self.cp_len)+1]; # Extract received OFDM Symbols
        print(f'Payload data shape : {payload_vec.shape}')
        payload_mat = np.matlib.reshape(payload_vec, (self.NoOFDMSymbols,(self.No_SC+self.cp_len) )).transpose(); # Reshape symbols
        #payload_mat = np.transpose(payload_mat)
                
        # Remove the cyclic prefix, keeping FFT_OFFSET samples of CP (on average)
        payload_mat_noCP = payload_mat[self.cp_len-self.FFT_OFFSET:self.cp_len-self.FFT_OFFSET+self.No_SC, :];
                
        # Take the FFT
        syms_f_mat = np.fft.fft(payload_mat_noCP, self.No_SC, axis=0);
                
        # Equalize (zero-forcing, just divide by complex chan estimates)
        rep_rx_H = np.transpose(np.matlib.repmat(H_est, self.NoOFDMSymbols,1))
        print(syms_f_mat.shape,rep_rx_H.shape )
        rx_syms = syms_f_mat / (rep_rx_H)
        #rx_syms = syms_f_mat
        return rx_syms
    def equalizeSymbols(self,samples,payload_index,channelEstimates):
        """
        Equalizing the frequency domain received signal with channel estimates
        Input:
            samples - time domain received signal after CFo correction
            payload_index - starting index of the payload samples
            channelEstimates - vector of complex channel gains
        Output:
            rx_syms - Equalized, frequency domain received data symbols
        """
        rx_dec_cfo_corr = samples[0,:]
        self.N_OFDM_SYMS, self.CP_LEN,self.N_SC = self.NoOFDMSymbols,self.cp_len,self.No_SC
        #print(f'Received data shape : {rx_dec_cfo_corr.shape}')
        payload_vec = rx_dec_cfo_corr[payload_index+1: payload_index+self.N_OFDM_SYMS*(self.N_SC+self.CP_LEN)+1]; # Extract received OFDM Symbols
        #print(f'Payload data shape : {payload_vec.shape}')
        payload_mat = np.matlib.reshape(payload_vec, (self.N_OFDM_SYMS,(self.N_SC+self.CP_LEN) )).transpose(); # Reshape symbols
        #payload_mat = np.transpose(payload_mat)
                
        # Remove the cyclic prefix, keeping FFT_OFFSET samples of CP (on average)
        payload_mat_noCP = payload_mat[self.CP_LEN-self.FFT_OFFSET:self.CP_LEN-self.FFT_OFFSET+self.N_SC, :];
                
        # Take the FFT
        syms_f_mat = np.fft.fft(payload_mat_noCP, self.N_SC, axis=0);
                
        # Equalize (zero-forcing, just divide by complex chan estimates)
        rep_rx_H = np.transpose(np.matlib.repmat(channelEstimates, self.N_OFDM_SYMS,1))
        rx_syms = syms_f_mat / (rep_rx_H)
        return rx_syms
    
    def demodumlate(self,equalizedSymbols):
        """
        Demodulating QAM symbols
        Input:
            equalizedSymbols - Equalized, frequency domain received signal
        Output:
            rx_data -received binary data
            rx_syms - Equalized, frequency domain received data symbols
        """
        payload_syms_mat = equalizedSymbols[self.SC_IND_DATA, :];
        ## Demodulate
        rx_syms = payload_syms_mat.transpose().reshape(1,self.N_DATA_SYMS)
        rx_syms = rx_syms.reshape(-1,self.N_DATA_SYMS)
        if self.MOD_ORDER == 16:
            demod_fcn_16qam = lambda x: (8*(np.real(x)>0)) + (4*(abs(np.real(x))<0.6325)) \
                            + (2*(np.imag(x)>0)) + (1*(abs(np.imag(x))<0.6325));
            rx_data = np.array(list(map(demod_fcn_16qam,rx_syms)))
        elif self.MOD_ORDER == 4:
            demod_fcn_qpsk = lambda x: (2*(np.real(x)>0)) + (1*(np.imag(x)>0)) 
            rx_data = np.array(list(map(demod_fcn_qpsk,rx_syms)))
        #rx_data = arrayfun(demod_fcn_16qam, rx_syms);
        elif self.MOD_ORDER == 2:
            demod_fcn_bpsk = lambda x: (1*(np.real(x)>0)) 
            rx_data = np.array(list(map(demod_fcn_bpsk,rx_syms)))
        
        return rx_data,rx_syms
    
    
    
        
        
    
    
if __name__ == "__main__":
    from dependencies import Estimator
    check = Estimator(64,10,2)
    Fs = 10e6
    ofdm_obj = signal_generator(No_SC=64, NoOFDMSymbols=10, ModOrder= 2,factor=2)
    detector_obj = symbol_detector(No_SC=64, NoOFDMSymbols=10, ModOrder= 2)
    waveform_OFDM, data_symbols_time,pad_zero = ofdm_obj.generate_data_OFDM_symbol(no_zeros_to_pad=200)
    
    # plt.plot(np.real(waveform_OFDM[0]))
    #print(detector_obj.out())
    waveform_OFDM = np.fromfile('./rx_data_tawfik.dat',dtype=np.complex64)[100:100000].reshape(1,-1)
    print(waveform_OFDM.shape)
    
    rx_pad_zero = detector_obj.decimateSamples(waveform_OFDM,factor=2)
    lts_corr, no_pkts,payload_start_indices = detector_obj.detectPeaks(rx_pad_zero)
    print(payload_start_indices)
    output_samples, cfo_estimates = detector_obj.estimate_carrier_freq_offset(rx_pad_zero,payload_start_indices[0],0)
    print(f'Estimated LTS CFO : {cfo_estimates*(Fs/2)*1e-0} Hz')
    
    H_est = detector_obj.estimate_channel(output_samples,payload_start_indices[0])
   
   
    #equalized_symbols = check.equalizeSymbols(output_samples,payload_start_indices[0], np.ones((1,64)))
    equalized_symbols = detector_obj.equalizeSymbols(output_samples,payload_start_indices[0],H_est)
    print(equalized_symbols.shape)
    plt.plot(lts_corr)
    plt.show()


    print(rx_pad_zero.shape, pad_zero.shape)
   # plt.scatter(equalized_symbols.real,equalized_symbols.imag)
    #plt.plot(output_samples[0].real)
   # plt.show()
    '''

 
    plt.plot(np.real(pad_zero[0]))
    plt.plot(np.real(rx_pad_zero[0]))
    plt.show()
    '''
    
    
    
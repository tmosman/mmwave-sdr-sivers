#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:37:55 2024

@author: tmosman

"""
import sys
import os
import scipy.io as sc
import ast
from scipy import signal as sig
import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np


class Estimator:
    def __init__(self,numberSubCarriers,numberOFDMSymbols,modOrder):
        self.N_OFDM_SYMS = numberOFDMSymbols  # Number of OFDM symbols
        self.MOD_ORDER   = modOrder           # Modulation order (2/4/16 = BSPK/QPSK/16-QAM)
        if numberSubCarriers == 64:
            self.subcarriersIndex_64(numberSubCarriers)
        self.lts_time,self.all_preamble = self.preamble()
        self.LTS_CORR_THRESH = 0.70
        self.FFT_OFFSET = 4
        pass
    
    def subcarriersIndex_64(self,NSubCarriers):
        self.N_SC   = NSubCarriers     # Number of subcarriers
        self.CP_LEN = 16      # Cyclic prefix length
        self.SC_IND_PILOTS = np.array([7,21,43,57]);  # Pilot subcarrier indices
        self.SC_IND_DATA  = np.array([1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,
                                      18,19,20,22,23,24,25,26,38,39,40,41,42,
                                      44,45,46,47,48,49,50,51,52,53,54,55,56,
                                      58,59,60,61,62,63]);     # Data subcarrier indices
       
        self.N_DATA_SYMS = self.N_OFDM_SYMS * len(self.SC_IND_DATA)  # Number of data symbols (one per data-bearing subcarrier per OFDM symbol)  
        
    def preamble(self, zc = False):
        # STS
        sts_f = np.zeros((1,self.N_SC ),'complex')
        sts_f[0,0:27] = [0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,
                         1+1j,0,0,0,1+1j,0,0]
        sts_f[0,38:]= [0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,
                       0,0,0,1+1j,0,0,0];
        sts_t = np.fft.ifft(np.sqrt(13/6)*sts_f[0,:],64)
        sts_t = sts_t[0:16];
           
        # LTS 
        lts_t = np.zeros((1,self.N_SC ),'complex')
        lts_f = [0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,
                 1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,
                 -1,-1,1,1,-1,1,-1,1,1,1,1];
        lts_t[0,:] = np.fft.ifft(lts_f,64);
    
        # Generation of expected preamble
        if zc == True:
            zc_len = 1023
            idx = np.arange(zc_len)
            M_root = 7
            zc = np.exp(-1j * np.pi * M_root / zc_len * idx * (idx + 1)).reshape(1,-1)
            preamble = np.concatenate((zc[0,:],lts_t[0,32:],lts_t[0,:],lts_t[0,:]),axis=0);
        else:
            AGC = np.matlib.repmat(sts_t, 1, 30);
            preamble = np.concatenate((AGC[0,:],lts_t[0,32:],lts_t[0,:],lts_t[0,:]),axis=0);
            
        return lts_t,preamble
    
    def generateOFDM(self):
        ## Generate a payload of random integers
        #tx_data = randi(MOD_ORDER, 1, N_DATA_SYMS) - 1;
        tx_data = np.random.randint(0,self.MOD_ORDER, self.N_DATA_SYMS)
        #print(tx_data.shape)
        '''
        % Functions for data -> complex symbol mapping (like qammod, avoids comm toolbox requirement)
        % These anonymous functions implement the modulation mapping from IEEE 802.11-2012 Section 18.3.5.8
        modvec_bpsk   =  (1/sqrt(2))  .* [-1 1];
        modvec_16qam  =  (1/sqrt(10)) .* [-3 -1 +3 +1];
        modvec_64qam  =  (1/sqrt(43)) .* [-7 -5 -1 -3 +7 +5 +1 +3];
        	
        mod_fcn_bpsk  = @(x) complex(modvec_bpsk(1+x),0);
        mod_fcn_qpsk  = @(x) complex(modvec_bpsk(1+bitshift(x, -1)), modvec_bpsk(1+mod(x, 2)));
        mod_fcn_16qam = @(x) complex(modvec_16qam(1+bitshift(x, -2)), modvec_16qam(1+mod(x,4)));
        mod_fcn_64qam = @(x) complex(modvec_64qam(1+bitshift(x, -3)), modvec_64qam(1+mod(x,8)));
        
        
        	% Reshape the symbol vector to a matrix with one column per OFDM symbol
        tx_syms_mat = reshape(tx_syms, length(SC_IND_DATA), N_OFDM_SYMS);
        save("packet_syms_usrp_sub6.mat","tx_syms_mat")	
        	% Define the pilot tone values as BPSK symbols
        pilots = [1 1 -1 1].';
        	
        	% Repeat the pilots across all OFDM symbols
        pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);
        	
        %% IFFT
        	
        	% Construct the IFFT input matrix
        ifft_in_mat = zeros(N_SC, N_OFDM_SYMS);
        	
        	% Insert the data and pilot values; other subcarriers will remain at 0
        ifft_in_mat(SC_IND_DATA, :)   = tx_syms_mat;
        ifft_in_mat(SC_IND_PILOTS, :) = pilots_mat;
        	
        	%Perform the IFFT
        tx_payload_mat = ifft(ifft_in_mat, N_SC, 1);
        	
        	% Insert the cyclic prefix
        if(CP_LEN > 0)
        	  tx_cp = tx_payload_mat((end-CP_LEN+1 : end), :);
        	  tx_payload_mat = [tx_cp; tx_payload_mat];
        end
        	
        	% Reshape to a vector
        tx_payload_vec = reshape(tx_payload_mat, 1, numel(tx_payload_mat));
        	
        % Construct the full time-domain OFDM waveform
        tx_vec = [preamble tx_payload_vec];
        
        
        % Pad with zeros for transmission
        tx_vec_padded_A = [tx_vec zeros(1,50)];
        '''
        
        modvec_bpsk   =  (1/np.sqrt(2))  *  np.array([-1,1])
        modvec_16qam  =  (1/np.sqrt(10)) * np.array( [-3, -1, +3, +1])
        modvec_64qam  =  (1/np.sqrt(43)) *  np.array([-7 ,-5, -1, -3 ,+7 ,+5, +1 ,+3]) 
        #modvec_bpsk[1+x]
        data_out = []
        mod_fcn_bpsk  = lambda x: complex(modvec_bpsk[x],0)
        
        for data_sample in tx_data:
            data_out.append(mod_fcn_bpsk(data_sample))
        
        tx_syms = np.array(data_out)
        print(np.array(data_out).shape)
        tx_syms_mat = np.matlib.reshape(tx_syms, (len(self.SC_IND_DATA), self.N_OFDM_SYMS));
        # np.matlib.reshape(
        #h_letters = [data_out.append(mod_fcn_bpsk(data_sample)) for data_sample in tx_data]
       
        print(tx_syms_mat.shape)
        #print(mod_fcn_bpsk(tx_data[0]))
        return 1
    
    def detectPeak_Only(self,data):
        
        data = self.decimate2x(data,0) ## Edited
        
        flip_lts = np.fliplr(self.lts_time)
        lts_corr = abs(np.convolve(np.conj(flip_lts[0,:]),data[0,:], mode='full'))
        lts_peaks = np.argwhere(lts_corr > self.LTS_CORR_THRESH*np.max(lts_corr));
        
        
        [LTS1, LTS2] = np.meshgrid(lts_peaks,lts_peaks);
        [lts_second_peak_index,_] = np.where(LTS2-LTS1 == np.shape(self.lts_time)[1]);
        valid_peak_indices = lts_peaks[lts_second_peak_index].flatten()
        #print(lts_second_peak_index)
        
        if len(valid_peak_indices) >=2:
            select_peak = np.argmax(lts_second_peak_index)-1
            payload_ind = valid_peak_indices[select_peak]
            lts_ind = payload_ind-160;
            lts_ind = lts_ind+1
            print('Number of Packets Detected: ',len(valid_peak_indices))
            
        else:
            payload_ind,lts_ind = 0,0
            print('No Packet Detected !!!')
        numberofPackets = len(valid_peak_indices)
       
        return lts_corr,numberofPackets,valid_peak_indices,data
     
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
        flip_lts = np.fliplr(self.lts_time)
        lts_corr = abs(np.convolve(np.conj(flip_lts[0,:]),data[0,:], mode='full'))
        lts_peaks = np.argwhere(lts_corr > self.LTS_CORR_THRESH*np.max(lts_corr));
        
        #print(len(lts_peaks))
        [LTS1, LTS2] = np.meshgrid(lts_peaks,lts_peaks);
        [lts_second_peak_index,_] = np.where(LTS2-LTS1 == np.shape(self.lts_time)[1]);
        valid_peak_indices = lts_peaks[lts_second_peak_index].flatten()
        #print(lts_second_peak_index)
        
        if len(valid_peak_indices) >=2:
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
        numberofPackets = len(valid_peak_indices)
       
        return lts_corr,numberofPackets,valid_peak_indices
    
    def decimate2x(self,data,idx):
        """
        Down sample the received IQ samples by a factor of 2
        
        """
        # Read the captured IQ sample file, filter
        filter_coeff = sc.loadmat('./files_req/filter_array.mat')['interp_filt2']
        
       # data_conv = np.convolve(data_recv[0,:],h[0,:],'full').reshape(1,-1)
        filter_data = sig.upfirdn(filter_coeff[0,:],data[idx,:],down=2).reshape(1,-1)
        
        return filter_data
    
    def cfoEstimate(self,dataSamples, lts_index, do_cfo):
        """
        Estimate and correct CFO
        Input:
            dataSamples - time domain received signal before CFo correction
            lts_index - starting index of the lts samples
            do_cfo - Boolean argument True/False
            
        Output:
            cfo_samples_t - data samples after cfo correction;
        """
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
    
    def complexChannelGain(self,dataSamples,lts_index):
        """
        compute channel estimates from LTS sequences
        Input:
            dataSamples - time domain received signal after CFo correction
            lts_index - starting index of the lts samples
            
        Output:
            rx_H_est - vector of complex channel gains
        """
        
        rx_lts = dataSamples[0,lts_index : lts_index+160]
        rx_lts1 = rx_lts[-64+-self.FFT_OFFSET + 96:-64+-self.FFT_OFFSET +160];  # Check indexing
        rx_lts2 = rx_lts[-self.FFT_OFFSET+96:-self.FFT_OFFSET+160]
        #print(rx_lts1.shape,self.N_SC)
        if rx_lts1.shape[0] and rx_lts2.shape[0] == self.N_SC:
            rx_lts1_f, rx_lts2_f = np.fft.fft(rx_lts1), np.fft.fft(rx_lts2)
            # Calculate channel estimate from average of 2 training symbols
            rx_H_est = np.fft.fft(self.lts_time,64) * (rx_lts1_f + rx_lts2_f)/2
            
        else:
            rx_H_est = np.ones((1,64), dtype=np.complex64)

        return rx_H_est
    
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
    
    def sfo_correction(self, rxSig_freq_eq,ch_est):
        """
        Apply Sample Frequency Offset
        Input:
            rxSig_freq_eq - Equalized, frequency domain received signal
            pilot_sc      - Pilot subcarriers (indexes)
            pilots_matrix - Pilots in matrix form
            n_ofdm_syms   - Number of OFDM symbols
        Output:
            rxSig_freq_eq - Frequency domain signal after SFO correction

            pilots = [1 1 -1 1].';
            pilots_mat = repmat(pilots, 1, N_OFDM_SYMS);	
        """
        n_ofdm_syms = self.N_OFDM_SYMS
        pilot_sc = self.SC_IND_PILOTS
        pilot_syms= np.array([1,1, -1, 1])
        #pilots_matrix = np.transpose(np.vstack((pilot_syms,pilot_syms)))
        pilots_matrix = np.transpose(np.vstack([pilot_syms] * n_ofdm_syms))
        
        # Extract pilots and equalize them by their nominal Tx values
        pilot_freq = rxSig_freq_eq[pilot_sc, :]
        #print(pilot_freq.shape,pilots_matrix.shape)
        pilot_freq_corr = pilot_freq * pilots_matrix
        
        # Compute phase of every RX pilot
        pilot_phase = np.angle(np.fft.fftshift(pilot_freq_corr))
        pilot_phase_uw = np.unwrap(pilot_phase)
        
        # Slope of pilot phase vs frequency of OFDM symbol
        pilot_shift = np.fft.fftshift(pilot_sc)
        pilot_shift_diff = np.diff(pilot_shift)
        pilot_shift_diff_mod = np.remainder(pilot_shift_diff, 64).reshape(len(pilot_shift_diff), 1)
        pilot_delta = np.matlib.repmat(pilot_shift_diff_mod, 1, n_ofdm_syms)
        pilot_slope = np.mean(np.diff(pilot_phase_uw, axis=0) / pilot_delta, axis=0)
        
        # Compute SFO correction phases
        tmp = np.array(range(-32, 32)).reshape(len(range(-32, 32)), 1)
        tmp2 = tmp * pilot_slope
        pilot_phase_sfo_corr = np.fft.fftshift(tmp2, 1)
        pilot_phase_corr = np.exp(-1j * pilot_phase_sfo_corr)
        
        # Apply correction per symbol
        rxSig_freq_eq = rxSig_freq_eq * pilot_phase_corr
        ch_est = ch_est* np.transpose(np.mean(pilot_phase_corr,axis=1))
       
        return rxSig_freq_eq,ch_est
    
    def phase_correction(self, rxSig_freq_eq):
        """
        Apply Phase Correction
        Input:
            rxSig_freq_eq - Equalized, time domain received signal
            pilot_sc      - Pilot subcarriers (indexes)
            pilots_matrix - Pilots in matrix form
        Output:
            phase_error   - Computed phase error
        """
        # Extract pilots and equalize them by their nominal Tx values
        pilot_sc = self.SC_IND_PILOTS
        n_ofdm_syms = self.N_OFDM_SYMS
        pilot_syms= np.array([1,1, -1, 1])
        pilots_matrix = np.transpose(np.vstack([pilot_syms] * n_ofdm_syms))
        pilot_freq = rxSig_freq_eq[pilot_sc, :]
        pilot_freq_corr = pilot_freq * pilots_matrix
        
        # Calculate phase error for each symbol
        phase_error = np.angle(np.mean(pilot_freq_corr, axis=0))
        
        return phase_error

    def compute_BER(self,receivedSymbols,receivedData):
      
        if self.MOD_ORDER == 16:
            tx_data = sc.loadmat("./files_req/tx_data_usrp_mmWave_16qam_v3.mat")['tx_data']
            tx_syms =  sc.loadmat("./files_req/packet_syms_QAM_usrp_mmWave_16qam_v3.mat")['tx_syms']
        elif self.MOD_ORDER == 4:
            tx_data = sc.loadmat("./files_req/tx_data_usrp_mmWave_qpsk_v3.mat")['tx_data']
            tx_syms =  sc.loadmat("./files_req/packet_syms_QAM_usrp_mmWave_qpsk_v3.mat")['tx_syms']
        elif self.MOD_ORDER == 2:
            tx_data = sc.loadmat("./files_req/tx_data_usrp_mmWave_bpsk_v3.mat")['tx_data']
            tx_syms =  sc.loadmat("./files_req/packet_syms_QAM_usrp_mmWave_bpsk_v3.mat")['tx_syms']
        
        bit_errs = np.sum((tx_data^receivedData) != 0)
        rx_evm  = np.sqrt(np.sum((np.real(receivedSymbols) - np.real(tx_syms))**2 \
                                     + (np.imag(receivedSymbols) - np.imag(tx_syms))**2)/(len(self.SC_IND_DATA) * self.N_OFDM_SYMS));
        #print('\nResults:\n');
       # print(f'Num Bytes:  {self.N_DATA_SYMS * np.log2(self.MOD_ORDER) / 8 }\n' );
        #print(f'Sym Errors:  {sym_errs} (of { self.N_DATA_SYMS} total symbols)\n');
        print(f'Bit Errors:  {bit_errs} (of {self.N_DATA_SYMS * np.log2(self.MOD_ORDER)} total bits) BER: {bit_errs/(self.N_DATA_SYMS * np.log2(self.MOD_ORDER))}\n')
        print(f'The Receiver EVM is : {(rx_evm)*100}%')
        bit_errs_rate = bit_errs/(self.N_DATA_SYMS * np.log2(self.MOD_ORDER))
        return bit_errs_rate,tx_syms,bit_errs,rx_evm
    
    def rx_process(self,data,payload_ind,Fs):
        lts_ind = payload_ind-160
        lts_ind = lts_ind+1
        
        # CFO correction
        dataOutput,rx_cfo_est_lts = self.cfoEstimate(data, lts_ind,do_cfo=True)
        #print(rx_cfo_est_lts)
        print(f'Estimated LTS CFO ; {rx_cfo_est_lts*(Fs/2)*1e-3} kHz')
        
        
        # channel estimation
        Hest = self.complexChannelGain(dataOutput,lts_ind)
         
        # Equalization
        try:
            equalizeSymbols = self.equalizeSymbols(dataOutput, payload_ind, Hest)
        except ValueError:
            print('Not Equalized Properly !!')
            return -1,100,0,0
        
        # Apply SFO correction
        sfo_syms,Hest_sfo = self.sfo_correction(equalizeSymbols,Hest)
        
        # Appy Phase Error correction
        phase_error = self.phase_correction(equalizeSymbols)
        #print(f'Shape PE: {phase_error.shape}')
        
        phase_error_est = np.mean(np.diff(np.unwrap(phase_error)))/(4e-6*np.pi)
        print(f'Phased Error Residual CFO Est: {phase_error_est*1e-3} kHz')
        
        
        # Apply SFO + PE to the channel estimate, to correct residual offsets
        Hest = Hest_sfo* np.exp(-1j * np.mean(phase_error) )
        all_equalized_syms = sfo_syms * np.exp(-1j * phase_error) 
        all_equalized_syms = equalizeSymbols * np.exp(-1j * phase_error) 

        # Demodulation of QAM symbols
        receivedData,receivedSymbols = self.demodumlate(all_equalized_syms)
        #receivedData,receivedSymbols = self.demodumlate(equalizeSymbols)
        
        # Compute BERS
        ber,tx_syms,err,rx_evm = self.compute_BER(receivedSymbols, receivedData)
    
        return ber,err,receivedSymbols,tx_syms
    
    def transmitIQData(self, IQdataTX):
        numIQdataSamples = len(IQdataTX)
        self.clientControlRadio.sendall(b"transmitIQSamples "+str.encode(str(numIQdataSamples)))
        no_bits = 14
        inphase = (IQdataTX.real*(2**(no_bits-1)-1))
        inphase = inphase.astype('int16')
        quadrature = (IQdataTX.imag*(2**(no_bits-1)-1))
        quadrature = quadrature.astype('int16')

    

def inspect_IQ_samples(fig, ax, data_samples,numberOFDMSymbols,modOrder,Fs,stop_inspect=1):
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ack_rep = 'recap'
     #### Define parameters ####
    if modOrder == 2:
        data_pilot = np.fromfile('./files_req/OFDM_packet_upsample_x2_bpsk_v3.dat', dtype=np.complex64)
    elif modOrder == 4:
        data_pilot = np.fromfile('./files_req/OFDM_packet_upsample_x2_qpsk_v3.dat', dtype=np.complex64)
    elif modOrder == 16:
        data_pilot = np.fromfile('./files_req/OFDM_packet_upsample_x2_16qam_v3.dat', dtype=np.complex64)
    elif modOrder == 64:
        data_pilot = np.fromfile('./files_req/OFDM_packet_upsample_x2_64qam_v3.dat', dtype=np.complex64)
    subset_data = data_pilot.shape[0]*10   
    channelEstObj = Estimator(numberSubCarriers=64, numberOFDMSymbols=numberOFDMSymbols, modOrder=modOrder)
   
    lts_corr,numberofDecPckts,_,_ = channelEstObj.detectPeak_Only(data_samples[0:subset_data].reshape(1,-1))    # check the number of packets
    print(f'The total number of packets captured are : {numberofDecPckts}')

    

    if numberofDecPckts > 0:
        
        win_size = data_pilot.shape[0]*10
        #win_size = data_samples.shape[0]
        num_win = data_samples.shape[0]//win_size
        num_win =1
        vnum = 0
        anum = 0
        
        for q in range(0,num_win):
            data = data_samples[q*win_size:(q*win_size)+win_size]
            #data = data_samples
            lts_corr,numberofPackets,valid_peak_indices,downsample_output = channelEstObj.detectPeak_Only(data.reshape(1,-1))
            fig.suptitle(f' Window Iter.: {q+1}/{num_win}; --> Total Number of Packets Detected : {numberofPackets}')
            ax[0].clear()
            ax[0].plot(data.real,label = 'Real')
            ax[0].plot(data.imag, label = 'Imaginary')
            ax[0].set_title('Captured IQ Samples')
            ax[0].legend(loc='lower right',fontsize="7")
            if  numberofPackets>0 and stop_inspect == 1:
                
                for select_packet in range(0,numberofPackets-1): 
                    payload_ind = valid_peak_indices[select_packet]
                    pktBegin = payload_ind - 640
                    pktEnd = payload_ind + int((data_pilot.shape[0] - 1280)/2)
                    ber,err,receivedSymbols,tx_syms = channelEstObj.rx_process(downsample_output,payload_ind,Fs)
                    
                    ax[1].clear()
                    ax[1].set_title('Window of Packets Detected')
                    ax[1].set_ylabel('Normalized Correlation Values')
                    ax[1].set_xlabel('Samples Indices')
                    ax[1].plot(lts_corr/np.max(lts_corr),c= 'b', linewidth=0.2,alpha=0.5)
                    ax[1].set_ylim([0,1.0])
                    ax[1].axhline(y=channelEstObj.LTS_CORR_THRESH,xmin=0,xmax=len(lts_corr),c="red",linewidth=1,zorder=0,linestyle = '--',label='Threshold')
                    ax[1].scatter([payload_ind],[channelEstObj.LTS_CORR_THRESH],color='r', marker='D',edgecolors='black',label='Packet Beginning Index')
                    
                    
                    ax[2].clear()
                    ax[2].set_title(f'BER => {ber:.4f}, Error(s): {err}')
                    ax[2].set_ylabel('Imaginary Component [Q]')
                    ax[2].set_xlabel('Real Component [I]')
                    ax[2].set_ylim([-1.5,1.5])
                    ax[2].set_xlim([-1.5,1.5])
                    anum+=1
                    if err != 100 :
                        ax[2].scatter(receivedSymbols.real,receivedSymbols.imag, s=0.5,  marker='o',label='RX')
                        ax[2].scatter(tx_syms.real,tx_syms.imag, s=20,  marker='o',c='r',label='TX')
                        ax[2].legend(loc='upper right',fontsize="5")
                        ax[1].axvline(x=pktBegin,ymin=0,ymax=channelEstObj.LTS_CORR_THRESH-0.1,c="green",linewidth=3,zorder=1,linestyle = '-',alpha = 0.4,label = 'Window Valid Packet')
                        ax[1].legend(loc='upper right',fontsize="10")
                        ax[1].axvline(x=pktEnd,ymin=0,ymax=channelEstObj.LTS_CORR_THRESH-0.1,c="green",linewidth=3,zorder=1,linestyle = '-',alpha = 0.4)
                        plt.pause(1)
                        vnum+=1
                    else:
                        ax[2].text(-1.5,0, f'Corrupted Packet at no. {select_packet} : index {payload_ind}', fontsize = 15,bbox = dict(facecolor = 'red', alpha = 0.5))
                        ax[1].axvline(x=pktBegin,ymin=0,ymax=channelEstObj.LTS_CORR_THRESH-0.1,c="red",linewidth=3,zorder=1,linestyle = '-',alpha = 0.4,label = 'Window Invalid Packet')
                        ax[1].legend(loc='upper right',fontsize="10")
                        ax[1].axvline(x=pktEnd,ymin=0,ymax=channelEstObj.LTS_CORR_THRESH-0.1,c="red",linewidth=3,zorder=1,linestyle = '-',alpha = 0.4)
                        plt.pause(0.01)
                    if vnum > 20:
                        ack_rep = 'valid_capture'
                        stop_inspect = 0
                        break
            if stop_inspect == 0 or int(q) >= int(num_win/2):
                break
            #else:
            #    ack_rep = 'recap'
            
            #    break   
                
    else:
        ack_rep = 'recap'
        #print('')
        
                
     
    #print(f'Number of Valid Packets detected : {vnum} at success rate {(vnum/anum)*100} %')
    #print(f'The Expected  no. Packets is {expected_no_pkts} ; Base on Config.  At Success rate; {(vnum/expected_no_pkts)*100} %')    
    #print(f'The Actual  no. Packets is {actual_no_pkts} ; Based on Received IQ samples. At Success rate; {(vnum/actual_no_pkts)*100} %')
    return ack_rep,lts_corr

